//! Drum Pads: sixteen sample pads on MIDI notes 36–51 (pad 1 = note 36).
//!
//! Each pad plays its own sample once per hit (one-shot: note-offs are ignored), with tune, an
//! exponential decay, a 12 dB/oct low-pass, level, pan, reverse, start and end points, and a choke
//! group: a hit on a pad cuts the other pads of its group (as an open hi-hat is cut by a closed
//! one). A pad hit again cuts its own previous hit. Hits start on their exact sample, with the
//! velocity as gain. The host hands each pad its audio with [`Plugin::set_pad_sample`]; a pad
//! without one is silent.

use std::sync::Arc;

use super::synth::{Event, EventKind, EventQueue, MAX_EVENTS};
use crate::params::{Params, choice, lin, log, param_plumbing, toggle};
use crate::util::{frames_in, sane_channels, sane_sr};
use crate::{Category, ParamInfo, Plugin, PluginInfo, SampleSource, Unit, db_to_gain};

/// Pads, on notes [`FIRST_NOTE`] up.
pub const PADS: usize = 16;
/// Pad 1's MIDI note.
pub const FIRST_NOTE: u8 = 36;
/// Parameters per pad, in this order.
const PER_PAD: usize = 9;
const TUNE: usize = 0;
const DECAY: usize = 1;
const CUTOFF: usize = 2;
const LEVEL: usize = 3;
const PAN: usize = 4;
const REVERSE: usize = 5;
const START: usize = 6;
const END: usize = 7;
const CHOKE: usize = 8;

const CHOKES: &[&str] = &["Off", "1", "2", "3", "4"];
/// The decay at its maximum lets the sample ring out.
const DECAY_MAX: f32 = 10_000.0;
/// The low-pass at its maximum is off.
const CUTOFF_MAX: f32 = 20_000.0;

macro_rules! pad_params {
    ($($n:literal),*) => {
        [$(
            lin(concat!("pad", $n, "_tune"), concat!("Pad ", $n, " Tune"), -24.0, 24.0, 0.0, Unit::Semitones),
            log(concat!("pad", $n, "_decay"), concat!("Pad ", $n, " Decay"), 10.0, DECAY_MAX, DECAY_MAX, Unit::Ms),
            log(concat!("pad", $n, "_cutoff"), concat!("Pad ", $n, " Low-pass"), 40.0, CUTOFF_MAX, CUTOFF_MAX, Unit::Hz),
            lin(concat!("pad", $n, "_level"), concat!("Pad ", $n, " Level"), -48.0, 12.0, 0.0, Unit::Db),
            lin(concat!("pad", $n, "_pan"), concat!("Pad ", $n, " Pan"), -100.0, 100.0, 0.0, Unit::Percent),
            toggle(concat!("pad", $n, "_reverse"), concat!("Pad ", $n, " Reverse"), false),
            lin(concat!("pad", $n, "_start"), concat!("Pad ", $n, " Start"), 0.0, 100.0, 0.0, Unit::Percent),
            lin(concat!("pad", $n, "_end"), concat!("Pad ", $n, " End"), 0.0, 100.0, 100.0, Unit::Percent),
            choice(concat!("pad", $n, "_choke"), concat!("Pad ", $n, " Choke Group"), CHOKES, 0),
        )*]
    };
}

static PAD_PARAMS: [ParamInfo; PADS * PER_PAD] = pad_params!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

pub static PADS_INFO: PluginInfo = PluginInfo {
    id: "drum_pads",
    name: "Drum Pads",
    short_name: "Pads",
    category: Category::Instrument,
    params: &PAD_PARAMS,
    is_instrument: true,
    audiosuite: false,
};

const VOICES: usize = 32;
/// How long a cut (choke or retrigger) takes to fade out, in seconds.
const CUT_S: f32 = 0.003;
/// Below this the decay envelope counts as silent.
const SILENT: f32 = 1.0e-4;

/// A pad's settings, derived from its parameters.
#[derive(Debug, Clone, Copy)]
struct Pad {
    semis: f64,
    /// Per-frame decay multiplier (1 = none).
    decay_k: f32,
    /// Low-pass coefficients (a1, a2, a3), `None` when off.
    lp: Option<(f32, f32, f32)>,
    gain: f32,
    /// Left and right gains from the pan (balance: unity at the centre).
    pan: (f32, f32),
    reverse: bool,
    /// Start and end as fractions of the sample (start <= end).
    span: (f64, f64),
    choke: usize,
}

impl Pad {
    const fn new() -> Self {
        Pad { semis: 0.0, decay_k: 1.0, lp: None, gain: 1.0, pan: (1.0, 1.0), reverse: false, span: (0.0, 1.0), choke: 0 }
    }
}

#[derive(Debug, Clone, Copy)]
struct Voice {
    pad: usize,
    active: bool,
    /// Read position in sample frames.
    pos: f64,
    step: f64,
    /// Frames the voice plays between: forward from `lo`, or backward from `hi`.
    lo: f64,
    hi: f64,
    reverse: bool,
    gain: f32,
    env: f32,
    /// Fading out after a cut: the per-frame gain step down, 0 when not cut.
    cut: f32,
    cut_gain: f32,
    /// Low-pass state per side (TPT state-variable filter).
    lp: [[f32; 2]; 2],
}

impl Voice {
    const fn new() -> Self {
        Voice {
            pad: 0,
            active: false,
            pos: 0.0,
            step: 1.0,
            lo: 0.0,
            hi: 0.0,
            reverse: false,
            gain: 0.0,
            env: 0.0,
            cut: 0.0,
            cut_gain: 1.0,
            lp: [[0.0; 2]; 2],
        }
    }
}

/// One loaded pad sample: the audio, its frames and its rate.
#[derive(Clone)]
struct Loaded {
    audio: Arc<dyn SampleSource>,
    len: usize,
    rate: f32,
}

/// Sixteen sample pads.
pub struct DrumPads {
    p: Params,
    pads: [Pad; PADS],
    samples: [Option<Loaded>; PADS],
    sr: f32,
    ch: usize,
    voices: [Voice; VOICES],
    events: EventQueue,
}

impl Default for DrumPads {
    fn default() -> Self {
        Self::new()
    }
}

/// 4-point cubic interpolation of `x` at fractional frame `pos` (>= 0); outside is silence.
#[inline]
fn hermite(x: &[f32], pos: f64) -> f32 {
    let i = pos as usize;
    let t = (pos - i as f64) as f32;
    let at = |k: usize| x.get(k).copied().unwrap_or(0.0);
    let x0 = at(i);
    if t == 0.0 {
        return if x0.is_finite() { x0 } else { 0.0 };
    }
    let (xm1, x1, x2) = (at(i.saturating_sub(1)), at(i.saturating_add(1)), at(i.saturating_add(2)));
    let c1 = 0.5 * (x1 - xm1);
    let c2 = xm1 - 2.5 * x0 + 2.0 * x1 - 0.5 * x2;
    let c3 = 0.5 * (x2 - xm1) + 1.5 * (x0 - x1);
    let y = ((c3 * t + c2) * t + c1) * t + x0;
    if y.is_finite() { y } else { 0.0 }
}

impl DrumPads {
    pub fn new() -> Self {
        const NONE: Option<Loaded> = None;
        let mut d = Self {
            p: Params::new(&PAD_PARAMS),
            pads: [Pad::new(); PADS],
            samples: [NONE; PADS],
            sr: 48_000.0,
            ch: 2,
            voices: [Voice::new(); VOICES],
            events: EventQueue::new(),
        };
        d.update();
        d
    }

    fn update(&mut self) {
        let sr = self.sr;
        for (i, pad) in self.pads.iter_mut().enumerate() {
            let v = |k: usize| self.p.v(i * PER_PAD + k);
            let pct = |k: usize| f64::from(v(k) / 100.0).clamp(0.0, 1.0);
            let decay = v(DECAY);
            let decay_k = if decay >= DECAY_MAX || !decay.is_finite() {
                1.0
            } else {
                // −60 dB after `decay` ms.
                let frames = (decay / 1000.0 * sr).max(1.0);
                (0.001f32.ln() / frames).exp()
            };
            let fc = v(CUTOFF);
            let lp = (fc.is_finite() && fc < CUTOFF_MAX && fc < 0.45 * sr).then(|| {
                let g = (std::f32::consts::PI * fc / sr).tan();
                let k = std::f32::consts::SQRT_2;
                let a1 = 1.0 / (1.0 + g * (g + k));
                (a1, g * a1, g * g * a1)
            });
            let pan = (v(PAN) / 100.0).clamp(-1.0, 1.0);
            let (s, e) = (pct(START), pct(END));
            *pad = Pad {
                semis: f64::from(v(TUNE)),
                decay_k,
                lp,
                gain: db_to_gain(v(LEVEL)),
                pan: ((1.0 - pan).min(1.0), (1.0 + pan).min(1.0)),
                reverse: v(REVERSE) >= 0.5,
                span: (s.min(e), s.max(e)),
                choke: {
                    let c = v(CHOKE);
                    if c.is_finite() && c > 0.0 { c.round() as usize } else { 0 }
                },
            };
        }
    }

    /// Fades a voice out quickly.
    fn cut(v: &mut Voice, sr: f32) {
        if v.active && v.cut == 0.0 {
            v.cut = 1.0 / (CUT_S * sr).max(1.0);
        }
    }

    fn hit(&mut self, note: u8, vel: u8) {
        let Some(pad) = usize::from(note).checked_sub(usize::from(FIRST_NOTE)).filter(|p| *p < PADS) else { return };
        if vel == 0 {
            return;
        }
        let (Some(s), Some(settings)) = (self.samples.get(pad).and_then(Option::as_ref), self.pads.get(pad).copied()) else { return };
        let (len, rate) = (s.len, s.rate);
        let step = f64::from(rate) / f64::from(self.sr) * 2f64.powf(settings.semis / 12.0);
        if len == 0 || !step.is_finite() || step <= 0.0 {
            return;
        }
        // Whole frames, so 60% of 1000 frames ends at frame 600 exactly.
        let (lo, hi) = ((settings.span.0 * len as f64).round(), (settings.span.1 * len as f64).round());
        if hi - lo < 1.0 {
            return;
        }
        // The pad's previous hit, and the other pads of its choke group, are cut.
        let sr = self.sr;
        let pads = self.pads;
        for v in self.voices.iter_mut().filter(|v| v.active) {
            let group = pads.get(v.pad).map_or(0, |p| p.choke);
            if v.pad == pad || (settings.choke > 0 && group == settings.choke) {
                Self::cut(v, sr);
            }
        }
        let idx = self.voices.iter().position(|v| !v.active).or_else(|| {
            // All busy: take the quietest.
            self.voices
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| (a.env * a.cut_gain).partial_cmp(&(b.env * b.cut_gain)).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
        });
        if let Some(v) = idx.and_then(|i| self.voices.get_mut(i)) {
            *v = Voice {
                pad,
                active: true,
                pos: if settings.reverse { hi - 1.0 } else { lo },
                step: step.min(256.0),
                lo,
                hi,
                reverse: settings.reverse,
                gain: f32::from(vel.min(127)) / 127.0 * settings.gain,
                env: 1.0,
                cut: 0.0,
                cut_gain: 1.0,
                lp: [[0.0; 2]; 2],
            };
        }
    }

    fn apply(&mut self, kind: EventKind) {
        match kind {
            EventKind::On(n, v) => self.hit(n, v),
            EventKind::AllOff => {
                let sr = self.sr;
                self.voices.iter_mut().for_each(|v| Self::cut(v, sr));
            }
            // One-shot pads ignore note-offs.
            EventKind::Off(_) | EventKind::Sustain(_) | EventKind::Bend(_) | EventKind::ResetControllers => {}
        }
    }

    /// Adds every voice's frame `n` into `io` and advances the voices.
    #[inline]
    fn render_frame(&mut self, io: &mut [Vec<f32>], ch: usize, n: usize) {
        for v in &mut self.voices {
            if !v.active {
                continue;
            }
            let (Some(s), Some(pad)) = (self.samples.get(v.pad).and_then(Option::as_ref), self.pads.get(v.pad)) else {
                v.active = false;
                continue;
            };
            let nch = s.audio.num_channels();
            let read = |c: usize| s.audio.channel(if nch == 1 { 0 } else { c }).map_or(0.0, |d| hermite(d, v.pos));
            let mut lr = [read(0), read(1)];
            if let Some((a1, a2, a3)) = pad.lp {
                for (x, st) in lr.iter_mut().zip(v.lp.iter_mut()) {
                    let v3 = *x - st[1];
                    let v1 = a1 * st[0] + a2 * v3;
                    let v2 = st[1] + a2 * st[0] + a3 * v3;
                    st[0] = 2.0 * v1 - st[0];
                    st[1] = 2.0 * v2 - st[1];
                    *x = if v2.is_finite() { v2 } else { 0.0 };
                }
            }
            let a = v.gain * v.env * v.cut_gain;
            let (l, r) = (lr[0] * a * pad.pan.0, lr[1] * a * pad.pan.1);
            match io.get_mut(..ch) {
                Some([mono]) => {
                    if let Some(x) = mono.get_mut(n) {
                        *x += 0.5 * (l + r);
                    }
                }
                Some(outs) => {
                    for (c, b) in outs.iter_mut().enumerate() {
                        if let Some(x) = b.get_mut(n) {
                            *x += if c % 2 == 0 { l } else { r };
                        }
                    }
                }
                None => {}
            }
            // Advance.
            v.env *= pad.decay_k;
            if v.cut > 0.0 {
                v.cut_gain -= v.cut;
            }
            v.pos = if v.reverse { v.pos - v.step } else { v.pos + v.step };
            let ended = if v.reverse { v.pos < v.lo } else { v.pos >= v.hi };
            if ended || !v.pos.is_finite() || v.env < SILENT || v.cut_gain <= 0.0 {
                v.active = false;
            }
        }
    }
}

impl Plugin for DrumPads {
    param_plumbing!(PADS_INFO);

    fn prepare(&mut self, sample_rate: f32, _max_block: usize, channels: usize) {
        self.sr = sane_sr(sample_rate);
        self.ch = sane_channels(channels);
        self.update();
        self.reset();
    }

    fn reset(&mut self) {
        for v in &mut self.voices {
            *v = Voice::new();
        }
        self.events.clear();
    }

    fn process(&mut self, io: &mut [Vec<f32>], frames: usize) {
        let ch = self.ch.min(io.len());
        let frames = frames_in(io, frames, ch);
        for b in io.iter_mut().take(ch) {
            b.iter_mut().take(frames).for_each(|x| *x = 0.0);
        }
        let mut evs = [Event { offset: 0, kind: EventKind::AllOff }; MAX_EVENTS];
        let count = {
            let sorted = self.events.sorted();
            for (d, s) in evs.iter_mut().zip(sorted) {
                *d = *s;
            }
            sorted.len().min(MAX_EVENTS)
        };
        self.events.clear();
        let mut next = 0;
        for n in 0..frames {
            while let Some(e) = evs.get(next).filter(|_| next < count) {
                if e.offset > n && n + 1 < frames {
                    break;
                }
                let kind = e.kind;
                self.apply(kind);
                next += 1;
            }
            self.render_frame(io, ch, n);
        }
        // Events for an empty block take effect immediately.
        for e in evs.iter().take(count).skip(next) {
            self.apply(e.kind);
        }
    }

    fn tail_samples(&self) -> usize {
        // The longest pad played two octaves down.
        let longest = self.samples.iter().flatten().map(|s| s.len as f32 * self.sr / s.rate.max(1.0) * 4.0).fold(0.0f32, f32::max);
        if longest.is_finite() { longest.min(1.0e9) as usize } else { 0 }
    }

    fn note_on(&mut self, offset: usize, note: u8, velocity: u8) {
        self.events.push(offset, EventKind::On(note, velocity));
    }

    fn note_off(&mut self, offset: usize, note: u8) {
        self.events.push(offset, EventKind::Off(note));
    }

    fn all_notes_off(&mut self) {
        self.events.push(0, EventKind::AllOff);
    }

    fn set_pad_sample(&mut self, pad: usize, sample: Option<Arc<dyn SampleSource>>) -> bool {
        let Some(slot) = self.samples.get_mut(pad) else { return false };
        for v in self.voices.iter_mut().filter(|v| v.pad == pad) {
            v.active = false;
        }
        *slot = sample.map(|audio| {
            let len = if audio.num_channels() == 0 { 0 } else { audio.frames() };
            let rate = audio.sample_rate();
            // An unusable rate plays at the host rate rather than not at all.
            let rate = if rate.is_finite() && rate > 0.0 { rate.min(1_536_000.0) } else { self.sr };
            Loaded { audio, len, rate }
        });
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SampleBuffer;

    const SR: f32 = 48_000.0;

    fn pads() -> DrumPads {
        let mut d = DrumPads::new();
        d.prepare(SR, 512, 2);
        d
    }

    fn load(d: &mut DrumPads, pad: usize, data: Vec<f32>) {
        assert!(d.set_pad_sample(pad, Some(Arc::new(SampleBuffer { sample_rate: SR, channels: vec![data] }))));
    }

    /// Renders `len` frames in `block`-frame blocks; returns the left channel.
    fn run(d: &mut DrumPads, len: usize, block: usize) -> Vec<f32> {
        let mut out = Vec::new();
        let mut buf = vec![vec![1.0f32; block]; 2];
        while out.len() < len {
            let n = block.min(len - out.len());
            d.process(&mut buf, n);
            out.extend_from_slice(&buf[0][..n]);
        }
        out
    }

    #[test]
    fn a_hit_starts_on_its_exact_sample_with_the_velocity_as_gain() {
        let mut d = pads();
        load(&mut d, 0, vec![0.5; 2000]);
        d.note_on(300, FIRST_NOTE, 127);
        let out = run(&mut d, 1024, 512);
        assert_eq!(out[299], 0.0);
        assert!((out[300] - 0.5).abs() < 1e-6, "{}", out[300]);
        // In a later block, at another offset, at half velocity.
        let mut d = pads();
        load(&mut d, 3, vec![0.5; 2000]);
        run(&mut d, 512, 512);
        d.note_on(77, FIRST_NOTE + 3, 64);
        let out = run(&mut d, 512, 512);
        assert_eq!(out[76], 0.0);
        assert!((out[77] - 0.5 * 64.0 / 127.0).abs() < 1e-6);
    }

    #[test]
    fn notes_outside_the_pads_and_empty_pads_are_silent() {
        let mut d = pads();
        load(&mut d, 0, vec![0.5; 2000]);
        d.note_on(0, FIRST_NOTE - 1, 127);
        d.note_on(0, FIRST_NOTE + PADS as u8, 127);
        d.note_on(0, FIRST_NOTE + 1, 127);
        assert!(run(&mut d, 1024, 256).iter().all(|x| *x == 0.0));
        assert!(!d.set_pad_sample(PADS, None), "no pad 17");
    }

    #[test]
    fn a_pad_plays_once_to_its_end() {
        let mut d = pads();
        load(&mut d, 0, vec![0.5; 1000]);
        d.note_on(0, FIRST_NOTE, 127);
        // A note-off changes nothing.
        d.note_off(10, FIRST_NOTE);
        let out = run(&mut d, 1500, 256);
        assert!((out[999] - 0.5).abs() < 1e-6);
        assert_eq!(out[1000], 0.0);
    }

    #[test]
    fn reverse_plays_backwards_and_start_end_trim() {
        let ramp: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        let mut d = pads();
        load(&mut d, 0, ramp.clone());
        d.set_param("pad1_reverse", 1.0);
        d.note_on(0, FIRST_NOTE, 127);
        let out = run(&mut d, 1200, 512);
        for i in 0..1000 {
            assert!((out[i] - ramp[999 - i]).abs() < 1e-6, "frame {i}");
        }
        assert_eq!(out[1000], 0.0);
        // Start 50%, end 60%: frames 500..600 only.
        let mut d = pads();
        load(&mut d, 0, ramp.clone());
        d.set_param("pad1_start", 50.0);
        d.set_param("pad1_end", 60.0);
        d.note_on(0, FIRST_NOTE, 127);
        let out = run(&mut d, 200, 200);
        assert!((out[0] - 0.5).abs() < 1e-6 && (out[99] - 0.599).abs() < 1e-6);
        assert_eq!(out[100], 0.0);
    }

    #[test]
    fn a_choke_group_cuts_its_other_pads() {
        let mut d = pads();
        load(&mut d, 0, vec![0.5; 48_000]);
        load(&mut d, 1, vec![0.0; 48_000]);
        d.set_param("pad1_choke", 1.0);
        d.set_param("pad2_choke", 1.0);
        d.note_on(0, FIRST_NOTE, 127);
        let out = run(&mut d, 1000, 500);
        assert!(out[999] > 0.4);
        d.note_on(0, FIRST_NOTE + 1, 127);
        let out = run(&mut d, 1000, 500);
        let fade = (CUT_S * SR) as usize + 1;
        assert!(out[0] > 0.0 && out[0] <= 0.5, "fades, doesn't click off");
        assert!(out[fade..].iter().all(|x| *x == 0.0), "cut within {fade} frames");
        // Without a group the pads overlap.
        let mut d = pads();
        load(&mut d, 0, vec![0.5; 48_000]);
        load(&mut d, 1, vec![0.25; 48_000]);
        d.note_on(0, FIRST_NOTE, 127);
        d.note_on(0, FIRST_NOTE + 1, 127);
        let out = run(&mut d, 100, 100);
        assert!((out[50] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn a_retrigger_cuts_the_pads_previous_hit() {
        let mut d = pads();
        load(&mut d, 0, vec![0.5; 48_000]);
        d.note_on(0, FIRST_NOTE, 127);
        d.note_on(100, FIRST_NOTE, 127);
        let out = run(&mut d, 1000, 1000);
        // One hit's level once the first has faded, not two stacked.
        assert!((out[900] - 0.5).abs() < 1e-6, "{}", out[900]);
    }

    #[test]
    fn tune_decay_low_pass_level_and_pan_shape_the_hit() {
        let ramp: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        // An octave up plays twice as fast.
        let mut d = pads();
        load(&mut d, 0, ramp);
        d.set_param("pad1_tune", 12.0);
        d.note_on(0, FIRST_NOTE, 127);
        let out = run(&mut d, 600, 600);
        assert!((out[250] - 0.5).abs() < 1e-3 && out[500] == 0.0);
        // A 100 ms decay: −60 dB at 100 ms.
        let mut d = pads();
        load(&mut d, 0, vec![1.0; 48_000]);
        d.set_param("pad1_decay", 100.0);
        d.note_on(0, FIRST_NOTE, 127);
        let out = run(&mut d, 4800, 512);
        assert!((out[4799] - 0.001).abs() < 1e-4, "{}", out[4799]);
        // The low-pass takes the top off.
        let hiss: Vec<f32> = (0..9600).map(|i| if i % 2 == 0 { 0.5 } else { -0.5 }).collect();
        let mut d = pads();
        load(&mut d, 0, hiss);
        d.set_param("pad1_cutoff", 500.0);
        d.note_on(0, FIRST_NOTE, 127);
        let out = run(&mut d, 9600, 512);
        assert!(out[4800..].iter().all(|x| x.abs() < 0.01));
        // −6 dB, panned hard left.
        let mut d = pads();
        load(&mut d, 0, vec![0.5; 100]);
        d.set_param("pad1_level", -6.0);
        d.set_param("pad1_pan", -100.0);
        d.note_on(0, FIRST_NOTE, 127);
        let mut buf = vec![vec![0.0f32; 64]; 2];
        d.process(&mut buf, 64);
        assert!((buf[0][10] - 0.5 * db_to_gain(-6.0)).abs() < 1e-6);
        assert_eq!(buf[1][10], 0.0);
    }

    #[test]
    fn hostile_values_stay_silent_or_finite() {
        let mut d = pads();
        load(&mut d, 0, vec![f32::NAN, f32::INFINITY, 0.5, 0.5]);
        assert!(d.set_pad_sample(1, Some(Arc::new(SampleBuffer { sample_rate: f32::NAN, channels: vec![vec![0.5; 10]] }))));
        assert!(d.set_pad_sample(2, Some(Arc::new(SampleBuffer { sample_rate: SR, channels: vec![] }))));
        for p in 0..PADS {
            d.set_param(&format!("pad{}_tune", p + 1), f32::NAN);
            d.note_on(0, FIRST_NOTE + p as u8, 127);
        }
        assert!(run(&mut d, 1024, 128).iter().all(|x| x.is_finite()));
    }
}
