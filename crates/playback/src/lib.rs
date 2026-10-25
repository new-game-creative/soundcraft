//! Realtime playback through the system audio device (cpal: CoreAudio, WASAPI, ALSA/Pulse/
//! PipeWire via ALSA, JACK, WebAudio).
//!
//! The UI thread owns a [`Player`]; the audio callback owns the [`MixEngine`] and a snapshot of
//! the session (`Arc<Session>`) that the UI swaps whenever the document changes. Commands go to
//! the callback through a channel; position and meters come back through atomics and a
//! `try_lock`ed snapshot, so the callback never blocks. When no device is available the player
//! falls back to a silent clock thread so the transport, meters and automation still run.
//!
//! Surround: the device is opened with as many output channels as it offers, up to the width of
//! the session's main format. Main channels (SMPTE/WAV order) go to device channels in order; when
//! the device has fewer channels than the main mix, surround is folded down to stereo (ITU-R
//! BS.775). A mono device receives the average of the stereo channels, before output clipping.
//!
//! Plugins and threads: hosted third-party plugins (CLAP/VST3) are created, given their stored
//! state and prepared on the thread that calls [`Player::update_session`] (the UI thread), and
//! travel to the audio thread together with the session snapshot that needs them; the mix
//! engine adopts them (see `soundcraft_mix`'s crate docs). Instances the engine retires, and the
//! session snapshots the callback replaces, come back on return channels and are dropped on the UI
//! thread ([`Player::idle`]). Plugin editors are driven, and plugin states for saving read
//! ([`Player::capture_states`]), on the UI thread
//! through handles taken from each instance before it leaves ([`Player::open_editor`]).
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod devices;
#[cfg(not(target_arch = "wasm32"))]
mod midi_in;
pub mod record;

pub use devices::AudioConfig;

use soundcraft_dsp::meter::LoudnessMeter;
use soundcraft_dsp::{Plugin, PluginEditor, PluginStateReader};
use soundcraft_mix::{MAX_CHANNELS, MixEngine, PreparedInstance, StripMeter, itu_stereo_fold, main_channels};
use soundcraft_model::{Session, TrackId};
use soundcraft_time::{Range, Samples};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, Ordering};
use std::sync::mpsc::{Receiver, Sender, SyncSender, channel, sync_channel};
use std::sync::{Arc, Mutex, PoisonError};

const BLOCK: usize = 512;

/// Room for replaced snapshots on their way back; `update_session` empties it before each send.
const OLD_SESSIONS: usize = 16;

/// The realtime audio thread marker lives in the mix engine, which carries it into the strips it
/// renders on worker threads; the cpal callbacks here set it.
pub use soundcraft_mix::{mark_audio_thread, on_audio_thread};

/// Meter values for the UI.
#[derive(Debug, Clone, Default)]
pub struct MeterSnapshot {
    pub tracks: HashMap<TrackId, StripMeter>,
    pub main: StripMeter,
    /// Loudness of the main mix during playback.
    pub loudness: Loudness,
}

/// ITU-R BS.1770 / EBU R128 loudness of the main mix, measured live while playing. It restarts
/// when playback starts and holds its last values after a stop. −144 means "not measured yet".
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Loudness {
    /// LUFS over the last 400 ms.
    pub momentary: f32,
    /// LUFS over the last 3 s.
    pub short_term: f32,
    /// Gated LUFS since playback started.
    pub integrated: f32,
    /// Highest true peak (4× oversampled) since playback started, dBTP.
    pub true_peak_db: f32,
}

impl Default for Loudness {
    fn default() -> Self {
        Loudness { momentary: -144.0, short_term: -144.0, integrated: -144.0, true_peak_db: -144.0 }
    }
}

impl Loudness {
    fn read(m: &LoudnessMeter) -> Self {
        Loudness { momentary: m.momentary_lufs(), short_term: m.short_term_lufs(), integrated: m.integrated_lufs(), true_peak_db: m.true_peak_db() }
    }
}

/// Plugin states read from the live instances: `(track, slot or INSTRUMENT_SLOT, state)`.
pub type PluginStates = Vec<(TrackId, usize, Vec<u8>)>;

/// A parameter edit made in a plugin's own editor: `(track, slot, param id, value)`.
pub type EditorEdit = (TrackId, usize, String, f32);

enum Cmd {
    /// A new snapshot plus the third-party instances it needs (created on the UI thread).
    Session(Arc<Session>, Vec<PreparedInstance>),
    /// Where the callback sends the snapshots it replaces (freeing one takes thousands of frees).
    OldSessions(SyncSender<Arc<Session>>),
    Input(Arc<record::InputRing>),
    Recording(bool),
    Play {
        from: Samples,
        end: Option<Samples>,
        looped: Option<Range>,
    },
    Stop,
    Locate(Samples),
    /// Launch `scene` on `track` (or stop it, `None`) at the next multiple of `quantize_bars`.
    Launch {
        track: TrackId,
        scene: Option<usize>,
        quantize_bars: f64,
        /// Continue from the replaced slot's loop position.
        legato: bool,
    },
    /// Tracks play their launched slots (on) or the timeline (off).
    SessionMode(bool),
    /// Plays `scene` on `track` now, as if launched at `start` (keeps a loop's phase across a
    /// transport restart).
    Resume {
        track: TrackId,
        scene: usize,
        start: Samples,
    },
    /// Launch `scene` on `track` at the end of the playing slot's current loop pass (or at the
    /// next multiple of `quantize_bars` when the track plays nothing).
    LaunchNext {
        track: TrackId,
        scene: usize,
        quantize_bars: f64,
    },
    /// Takes recording inside the mix (`None`: stop feeding them).
    Taps(Option<Arc<record::TapFeed>>),
    /// A live MIDI message from input `device` (an index into [`Cmd::MidiDevices`], or
    /// [`KEYBOARD`]).
    Midi {
        device: u16,
        data: [u8; 3],
    },
    /// A note played straight into `track` (an on-screen keyboard or pad), whatever its input.
    TrackMidi {
        track: TrackId,
        data: [u8; 3],
    },
    /// Audition a sound over the mix (`None`: stop).
    Preview(Option<Arc<PreviewSound>>),
    /// The names of the open MIDI inputs, by device index.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    MidiDevices(Arc<Vec<String>>),
}

/// A sound to audition (a file in a browser, say): planar samples at `rate`, played once over
/// the mix, also while the transport is stopped (see [`Player::preview`]).
#[derive(Debug, Clone, PartialEq)]
pub struct PreviewSound {
    pub channels: Vec<Vec<f32>>,
    pub rate: u32,
}

impl PreviewSound {
    fn frames(&self) -> usize {
        self.channels.iter().map(Vec::len).min().unwrap_or(0)
    }

    /// Frame `at` (fractional) of channel `c`, linearly interpolated.
    fn sample(&self, c: usize, at: f64) -> f32 {
        let Some(ch) = self.channels.get(c.min(self.channels.len().saturating_sub(1))) else { return 0.0 };
        let i = at.floor() as usize;
        let t = (at - at.floor()) as f32;
        let a = ch.get(i).copied().unwrap_or(0.0);
        let b = ch.get(i + 1).copied().unwrap_or(0.0);
        let v = a + (b - a) * t;
        if v.is_finite() { v } else { 0.0 }
    }
}

/// The MIDI "device" of the computer keyboard (see [`Player::play_midi`]).
pub const KEYBOARD: u16 = u16::MAX;
/// Played notes kept for the UI between its reads.
const PLAYED_CAPACITY: usize = 8192;

/// A live note played into a record-enabled track while recording: a note on, or a note off when
/// `velocity` is 0, at transport position `at`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlayedNote {
    pub track: TrackId,
    pub at: Samples,
    pub pitch: u8,
    pub velocity: u8,
}

/// Whether a track's MIDI input takes messages from `device` (by index into `names`).
fn takes_midi_from(input: &soundcraft_model::Route, device: u16, names: &[String]) -> bool {
    use soundcraft_model::Route;
    match input {
        // Every device, but not the computer keyboard: that is chosen on purpose (its keys are
        // shortcuts otherwise).
        Route::None => device != KEYBOARD,
        Route::Midi(want) if want.is_empty() => device != KEYBOARD,
        Route::Midi(want) if want == "keyboard" => device == KEYBOARD,
        Route::Midi(want) => names.get(usize::from(device)).is_some_and(|n| n == want),
        _ => false,
    }
}

/// What the clip launcher is playing and has queued, for the UI.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LaunchStatus {
    pub session_mode: bool,
    /// (track, scene, start sample) of every playing slot.
    pub playing: Vec<(TrackId, usize, Samples)>,
    /// (track, scene or `None` to stop, sample it happens at) of every queued change.
    pub queued: Vec<(TrackId, Option<usize>, Samples)>,
}

/// Most launches the audio thread keeps without allocating (one per track and a queued change).
const LAUNCH_CAPACITY: usize = 256;

impl LaunchStatus {
    /// Sized so the audio thread's copies into it never allocate.
    fn reserved() -> Self {
        LaunchStatus { session_mode: false, playing: Vec::with_capacity(LAUNCH_CAPACITY), queued: Vec::with_capacity(LAUNCH_CAPACITY) }
    }
}

struct Shared {
    position: AtomicI64,
    playing: AtomicBool,
    /// Speed × 1000.
    speed: AtomicU32,
    meters: Mutex<MeterSnapshot>,
    launch: Mutex<LaunchStatus>,
    /// Live MIDI played into record-enabled tracks while recording, for the UI to make clips of.
    played: Mutex<Vec<PlayedNote>>,
    /// A sound is being auditioned.
    previewing: AtomicBool,
}

/// What stopped the audio output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputTrouble {
    /// A block panicked; the output stays silent.
    Bug,
    /// The device went away.
    Gone,
    /// The device stopped asking for audio while playing.
    Stalled,
}

/// How the audio output is doing, for the UI.
#[derive(Default)]
struct Fault {
    /// A block panicked: output stays silent from then on.
    failed: AtomicBool,
    gone: AtomicBool,
    /// Blocks so far, to tell a device that stopped asking for audio.
    beats: std::sync::atomic::AtomicU64,
    /// The UI has been told.
    reported: AtomicBool,
}

impl Fault {
    /// What stopped the output, once.
    fn trouble(&self, stalled: bool) -> Option<OutputTrouble> {
        let trouble = if self.failed.load(Ordering::Relaxed) {
            OutputTrouble::Bug
        } else if self.gone.load(Ordering::Relaxed) {
            OutputTrouble::Gone
        } else if stalled {
            OutputTrouble::Stalled
        } else {
            return None;
        };
        (!self.reported.swap(true, Ordering::Relaxed)).then_some(trouble)
    }
}

/// How long a playing output may go without a block before it counts as stalled.
#[cfg(not(target_arch = "wasm32"))]
const STALL: std::time::Duration = std::time::Duration::from_secs(2);

/// True once the block count `beats` has stood still for `STALL` while playing; `seen` is the
/// count last seen and since when.
#[cfg(not(target_arch = "wasm32"))]
fn stalled(seen: &mut Option<(u64, std::time::Instant)>, beats: u64, playing: bool) -> bool {
    match *seen {
        Some((b, since)) if playing && b == beats => since.elapsed() >= STALL,
        _ => {
            *seen = Some((beats, std::time::Instant::now()));
            false
        }
    }
}

/// Fill one block with `fill`. If it panics, the block and every later one are silent and the
/// transport stops, rather than the panic taking the app down.
fn guarded_block(fault: &Fault, shared: &Shared, out: &mut [f32], fill: impl FnOnce(&mut [f32])) {
    fault.beats.fetch_add(1, Ordering::Relaxed);
    if fault.failed.load(Ordering::Relaxed) || std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| fill(out))).is_err() {
        fault.failed.store(true, Ordering::Relaxed);
        shared.playing.store(false, Ordering::Relaxed);
        out.fill(0.0);
    }
}

/// State owned by the audio callback.
struct AudioState {
    rx: Receiver<Cmd>,
    shared: Arc<Shared>,
    session: Arc<Session>,
    mix: MixEngine,
    pos: Samples,
    playing: bool,
    end: Option<Samples>,
    looped: Option<Range>,
    /// Rendered frames waiting to be consumed (planar, main channels) and the fractional read index.
    pending: Vec<Vec<f32>>,
    /// Stereo fold-down gains per main channel (used when the device has fewer channels).
    fold: [(f32, f32); MAX_CHANNELS],
    pending_len: usize,
    read: f64,
    device_rate: f64,
    input: Option<Arc<record::InputRing>>,
    recording: bool,
    /// Retired plugin instances go back to the UI thread to be dropped there.
    garbage: Sender<Vec<Box<dyn Plugin>>>,
    old_sessions: Option<SyncSender<Arc<Session>>>,
    /// Loudness of the main mix, sized to its channels (rebuilt only when the format changes).
    loudness: LoudnessMeter,
    /// Launcher changes waiting for their quantise boundary: (track, scene or stop, sample,
    /// legato).
    launch_queue: Vec<(TrackId, Option<usize>, Samples, bool)>,
    /// Takes recording inside the mix, fed after each block while recording.
    taps: Option<Arc<record::TapFeed>>,
    /// MIDI input names by device index.
    midi_names: Arc<Vec<String>>,
    /// The sound being auditioned and where it is (frames at its own rate).
    preview: Option<(Arc<PreviewSound>, f64)>,
}

fn new_engine(sr: f32) -> MixEngine {
    let mut m = MixEngine::new(sr, BLOCK);
    m.set_external_instances(true);
    m.metronome = true;
    m.tapped.reserve(64);
    m
}

impl AudioState {
    fn new(rx: Receiver<Cmd>, shared: Arc<Shared>, session: Arc<Session>, device_rate: f64, garbage: Sender<Vec<Box<dyn Plugin>>>) -> Self {
        let sr = session.sample_rate.as_f64() as f32;
        let nch = main_channels(&session);
        AudioState {
            garbage,
            old_sessions: None,
            fold: fold_gains(&session),
            rx,
            shared,
            mix: new_engine(sr),
            session,
            pos: 0,
            playing: false,
            end: None,
            looped: None,
            pending: vec![vec![0.0; BLOCK]; nch],
            pending_len: 0,
            read: 0.0,
            device_rate,
            input: None,
            recording: false,
            loudness: LoudnessMeter::new(sr, nch),
            launch_queue: Vec::with_capacity(LAUNCH_CAPACITY),
            taps: None,
            midi_names: Arc::new(Vec::new()),
            preview: None,
        }
        .with_launch_capacity()
    }

    /// Adds the next `n` frames of the auditioned sound to the block (resampled to the session
    /// rate; a mono sound plays on every channel). It ends on its own at its last frame.
    fn mix_preview(&mut self, n: usize) {
        let Some((sound, at)) = &mut self.preview else { return };
        let step = f64::from(sound.rate.max(1)) / self.session.sample_rate.as_f64().max(1.0);
        let frames = sound.frames() as f64;
        for i in 0..n {
            let pos = *at + i as f64 * step;
            if pos >= frames {
                break;
            }
            for (c, out) in self.pending.iter_mut().enumerate() {
                if let Some(x) = out.get_mut(i) {
                    *x += sound.sample(c, pos);
                }
            }
        }
        *at += n as f64 * step;
        if *at >= frames {
            // The UI keeps the sound, so it is not freed here.
            self.preview = None;
            self.shared.previewing.store(false, Ordering::Relaxed);
        }
    }

    /// Lists the tracks the taps record in the mix (within the reserved capacity: no allocation).
    fn sync_tapped(&mut self) {
        self.mix.tapped.clear();
        for tap in self.taps.iter().flat_map(|f| f.taps.iter()) {
            if let record::TapSource::Track(t) = tap.source
                && !self.mix.tapped.contains(&t)
                && self.mix.tapped.len() < self.mix.tapped.capacity()
            {
                self.mix.tapped.push(t);
            }
        }
    }

    /// Hands a live note to every instrument track listening to `device`, and while recording
    /// keeps it for the record-enabled ones. Other messages are ignored.
    fn route_midi(&mut self, device: u16, data: [u8; 3]) {
        let (status, pitch, vel) = (data[0] & 0xF0, data[1] & 0x7F, data[2] & 0x7F);
        let velocity = match status {
            0x90 => vel,
            0x80 => 0,
            _ => return,
        };
        let at = self.pos;
        let recording = self.recording && self.playing;
        let mut played = if recording { self.shared.played.try_lock().ok() } else { None };
        for t in self.session.tracks.iter().filter(|t| soundcraft_mix::plays_live_midi(t)) {
            if !takes_midi_from(&t.mixer.input, device, &self.midi_names) {
                continue;
            }
            if self.mix.live_notes.len() < soundcraft_mix::LIVE_NOTES {
                self.mix.live_notes.push(soundcraft_mix::LiveNote { track: t.id, offset: 0, pitch, velocity });
            }
            if t.mixer.record_arm
                && let Some(p) = played.as_mut()
                && p.len() < p.capacity()
            {
                p.push(PlayedNote { track: t.id, at, pitch, velocity });
            }
        }
    }

    /// Where the slot `track` plays ends its current loop pass, while the transport runs.
    fn pass_end(&self, track: TrackId) -> Option<Samples> {
        if !self.playing {
            return None;
        }
        let (scene, start) = self.mix.launch.playing(track)?;
        let slot = self.session.launcher.slot(track, scene)?;
        soundcraft_mix::launch_iteration(&self.session, slot, start, self.pos.max(start)).map(|(_, end)| end)
    }

    /// A note on or off for `track` alone, when it plays live MIDI (armed or monitoring).
    fn track_midi(&mut self, track: TrackId, data: [u8; 3]) {
        let (status, pitch, vel) = (data[0] & 0xF0, data[1] & 0x7F, data[2] & 0x7F);
        let velocity = match status {
            0x90 => vel,
            0x80 => 0,
            _ => return,
        };
        let plays = self.session.track(track).is_some_and(soundcraft_mix::plays_live_midi);
        if plays && self.mix.live_notes.len() < soundcraft_mix::LIVE_NOTES {
            self.mix.live_notes.push(soundcraft_mix::LiveNote { track, offset: 0, pitch, velocity });
        }
    }

    fn with_launch_capacity(mut self) -> Self {
        self.mix.launch.playing.reserve(LAUNCH_CAPACITY);
        self.mix.tapped.reserve(64);
        self
    }

    /// The first sample at or after `pos` on a multiple of `quantize_bars` bars (from bar 1).
    fn launch_time(&self, quantize_bars: f64) -> Samples {
        let s = &self.session;
        let sr = s.sample_rate;
        let tick = s.tempo.samples_to_ticks(self.pos, sr);
        let q = (quantize_bars * s.tempo.meter_at_tick(tick).ticks_per_bar() as f64).round() as i64;
        if q <= 0 || !self.playing {
            return self.pos;
        }
        let next = tick.div_euclid(q).saturating_add(i64::from(tick.rem_euclid(q) != 0)).saturating_mul(q);
        s.tempo.tick_to_samples(next, sr).max(self.pos)
    }

    /// Applies the queued launches due by `pos`, and returns how many frames may render before
    /// the next one (so each lands on its sample).
    fn run_launch_queue(&mut self, n: usize) -> usize {
        let pos = self.pos;
        let mut i = 0;
        while let Some(&(track, scene, at, legato)) = self.launch_queue.get(i) {
            if at <= pos {
                if legato {
                    self.mix.launch.set_legato(track, scene, pos);
                } else {
                    self.mix.launch.set(track, scene, pos);
                }
                self.launch_queue.swap_remove(i);
            } else {
                i += 1;
            }
        }
        // Every block, not only on changes: a publish that finds the UI holding the lock is
        // simply made again on the next one.
        self.publish_launch();
        let next = self.launch_queue.iter().map(|(_, _, at, _)| *at).min();
        next.map_or(n, |at| n.min(usize::try_from(at - pos).unwrap_or(n).max(1)))
    }

    /// Copies the launcher state for the UI (without allocating once the lists are sized).
    fn publish_launch(&self) {
        if let Ok(mut st) = self.shared.launch.try_lock() {
            st.session_mode = self.mix.launch.session_mode;
            st.playing.clear();
            st.playing.extend_from_slice(&self.mix.launch.playing);
            st.queued.clear();
            st.queued.extend(self.launch_queue.iter().map(|&(t, scene, at, _)| (t, scene, at)));
        }
    }

    fn drain(&mut self) {
        let mut changed = false;
        while let Ok(c) = self.rx.try_recv() {
            match c {
                Cmd::Input(r) => self.input = Some(r),
                Cmd::Recording(r) => self.recording = r,
                Cmd::Session(s, instances) => {
                    changed = true;
                    if (s.sample_rate.as_f64() - self.session.sample_rate.as_f64()).abs() > 0.5 {
                        let mut old = std::mem::replace(&mut self.mix, new_engine(s.sample_rate.as_f64() as f32));
                        let _ = self.garbage.send(old.drain_plugins());
                        self.sync_tapped();
                    }
                    self.mix.adopt(instances);
                    // The main format changed (a document edit, not steady state): resize.
                    let nch = main_channels(&s);
                    let sr = s.sample_rate.as_f64() as f32;
                    let resized = self.pending.len() != nch;
                    if resized {
                        self.pending = vec![vec![0.0; BLOCK]; nch];
                        self.pending_len = 0;
                        self.read = 0.0;
                    }
                    if resized || (self.loudness.sample_rate() - sr).abs() > 0.5 {
                        self.loudness = LoudnessMeter::new(sr, nch);
                    }
                    self.fold = fold_gains(&s);
                    let old = std::mem::replace(&mut self.session, s);
                    if let Some(tx) = &self.old_sessions {
                        let _ = tx.try_send(old);
                    }
                }
                Cmd::OldSessions(tx) => self.old_sessions = Some(tx),
                Cmd::Play { from, end, looped } => {
                    self.pos = from;
                    self.end = end;
                    self.looped = looped.filter(|r| r.len() > 64);
                    self.playing = true;
                    self.pending_len = 0;
                    self.read = 0.0;
                    self.mix.reset();
                    // Each playback pass is measured from its start.
                    self.loudness.reset();
                }
                Cmd::Stop => {
                    self.playing = false;
                    self.end = None;
                    self.looped = None;
                    self.mix.reset();
                    // Stopping the transport stops every launched slot.
                    self.mix.launch.playing.clear();
                    self.launch_queue.clear();
                    self.publish_launch();
                }
                Cmd::Launch { track, scene, quantize_bars, legato } => {
                    let at = self.launch_time(quantize_bars);
                    self.launch_queue.retain(|(t, _, _, _)| *t != track);
                    if self.launch_queue.len() < LAUNCH_CAPACITY {
                        self.launch_queue.push((track, scene, at, legato));
                    }
                    self.mix.launch.session_mode = true;
                    self.publish_launch();
                }
                Cmd::SessionMode(on) => {
                    self.mix.launch.session_mode = on;
                    if !on {
                        self.mix.launch.playing.clear();
                        self.launch_queue.clear();
                    }
                    self.publish_launch();
                }
                Cmd::Resume { track, scene, start } => {
                    self.launch_queue.retain(|(t, _, _, _)| *t != track);
                    self.mix.launch.set(track, Some(scene), start);
                    self.mix.launch.session_mode = true;
                    self.publish_launch();
                }
                Cmd::LaunchNext { track, scene, quantize_bars } => {
                    let at = self.pass_end(track).unwrap_or_else(|| self.launch_time(quantize_bars));
                    self.launch_queue.retain(|(t, _, _, _)| *t != track);
                    if self.launch_queue.len() < LAUNCH_CAPACITY {
                        self.launch_queue.push((track, Some(scene), at, false));
                    }
                    self.mix.launch.session_mode = true;
                    self.publish_launch();
                }
                Cmd::Locate(at) => {
                    self.pos = at;
                    self.pending_len = 0;
                    self.read = 0.0;
                    self.mix.reset();
                }
                Cmd::Taps(feed) => {
                    self.taps = feed;
                    self.sync_tapped();
                }
                Cmd::MidiDevices(names) => self.midi_names = names,
                Cmd::Preview(p) => {
                    self.preview = p.map(|p| (p, 0.0));
                    self.shared.previewing.store(self.preview.is_some(), Ordering::Relaxed);
                }
                Cmd::Midi { device, data } => self.route_midi(device, data),
                Cmd::TrackMidi { track, data } => self.track_midi(track, data),
            }
        }
        if changed {
            // Place adopted instances now (not only once playback starts), so editors see them;
            // `sync` creates no third-party plugins here.
            self.mix.ensure_synced(&self.session);
        }
        self.send_garbage();
        self.shared.playing.store(self.playing, Ordering::Relaxed);
    }

    fn send_garbage(&mut self) {
        let g = self.mix.take_retired();
        if !g.is_empty() {
            let _ = self.garbage.send(g);
        }
    }

    /// Render the next block at the session rate into `pending`.
    fn render_next(&mut self, quantum: usize) {
        let mut n = quantum.clamp(1, BLOCK);
        if let Some(l) = self.looped
            && self.pos >= l.end
        {
            self.pos = l.start;
        }
        if let Some(l) = self.looped {
            n = n.min(usize::try_from((l.end - self.pos).max(1)).unwrap_or(BLOCK));
        } else if let Some(e) = self.end {
            if self.pos >= e {
                self.playing = false;
                self.end = None;
                self.shared.playing.store(false, Ordering::Relaxed);
                n = 0;
            } else {
                n = n.min(usize::try_from(e - self.pos).unwrap_or(BLOCK));
            }
        }
        if n == 0 {
            for c in &mut self.pending {
                c.iter_mut().for_each(|x| *x = 0.0);
            }
            self.pending_len = BLOCK;
            return;
        }
        if self.playing {
            // Launches start exactly on their sample: the block ends at the next one.
            n = self.run_launch_queue(n);
        }
        if let Some(ring) = &self.input {
            let ch = ring.channels.load(Ordering::Relaxed).max(1);
            if self.mix.input.len() != ch {
                self.mix.input = vec![vec![0.0; BLOCK]; ch];
            }
            ring.pop_into(&mut self.mix.input, n, self.session.sample_rate.hz());
        }
        self.mix.monitor_only = !self.playing;
        self.mix.recording = self.recording;
        self.mix.render(&self.session, self.pos, n, &mut self.pending);
        self.mix.live_notes.clear();
        self.mix_preview(n);
        if self.recording
            && self.playing
            && let Some(feed) = &self.taps
        {
            for tap in &feed.taps {
                match tap.source {
                    record::TapSource::Track(t) => {
                        if let Some(src) = self.mix.tap(t) {
                            tap.push(src, n);
                        }
                    }
                    record::TapSource::Main => tap.push(&self.pending, n),
                }
            }
        }
        self.send_garbage();
        self.pending_len = n;
        if self.playing {
            self.loudness.process(&self.pending, n);
        }
        if let Ok(mut m) = self.shared.meters.try_lock() {
            m.main.copy_from(&self.mix.main_meter);
            if self.playing {
                m.loudness = Loudness::read(&self.loudness);
            }
            for (k, v) in &self.mix.meters {
                match m.tracks.get_mut(k) {
                    Some(d) => d.copy_from(v),
                    None => {
                        m.tracks.insert(*k, v.clone());
                    }
                }
            }
        }
        if self.playing {
            self.pos += n as i64;
            self.shared.position.store(self.pos, Ordering::Relaxed);
        }
    }

    /// Fill an interleaved device buffer.
    fn fill(&mut self, out: &mut [f32], channels: usize) {
        self.drain();
        let channels = channels.max(1);
        let monitoring = (self.input.is_some() && self.session.tracks.iter().any(|t| t.mixer.input_monitor || t.mixer.record_arm))
            || self.session.tracks.iter().any(soundcraft_mix::plays_live_midi)
            || self.preview.is_some();
        if !self.playing && !monitoring {
            out.iter_mut().for_each(|x| *x = 0.0);
            if let Ok(mut m) = self.shared.meters.try_lock() {
                m.main = StripMeter::default();
                m.tracks.clear();
            }
            self.publish_launch();
            return;
        }
        let speed = f64::from(self.shared.speed.load(Ordering::Relaxed)) / 1000.0;
        let step = self.session.sample_rate.as_f64() / self.device_rate.max(1.0) * speed.clamp(0.1, 4.0);
        let callback_frames = out.len() / channels;
        let quantum = ((callback_frames as f64 * step).ceil() as usize).clamp(1, BLOCK);
        for frame in out.chunks_mut(channels) {
            while self.read >= self.pending_len as f64 {
                self.read -= self.pending_len as f64;
                if !self.playing && !monitoring {
                    break;
                }
                self.render_next(quantum);
                if self.pending_len == 0 {
                    break;
                }
            }
            let i0 = self.read.floor() as usize;
            let fr = (self.read - i0 as f64) as f32;
            let sounding = self.playing || monitoring;
            let nch = self.pending.len();
            let sample = |c: usize| {
                let ch = self.pending.get(c);
                let a = ch.and_then(|v| v.get(i0)).copied().unwrap_or(0.0);
                let b = ch.and_then(|v| v.get(i0 + 1)).copied().unwrap_or(a);
                a + (b - a) * fr
            };
            if channels < nch {
                // Fold surround to stereo first; a mono device averages the two sides.
                let (mut l, mut r) = (0.0f32, 0.0f32);
                if sounding {
                    if nch == 2 {
                        (l, r) = (sample(0), sample(1));
                    } else {
                        for (c, (gl, gr)) in self.fold.iter().enumerate().take(nch) {
                            let v = sample(c);
                            l += v * gl;
                            r += v * gr;
                        }
                    }
                }
                for (c, o) in frame.iter_mut().enumerate() {
                    *o = match c {
                        0 if channels == 1 => {
                            let mono = (0.5 * l + 0.5 * r).clamp(-1.0, 1.0);
                            if mono.is_finite() { mono } else { 0.0 }
                        }
                        0 => l.clamp(-1.0, 1.0),
                        1 => r.clamp(-1.0, 1.0),
                        _ => 0.0,
                    };
                }
            } else {
                for (c, o) in frame.iter_mut().enumerate() {
                    *o = if c < nch.max(1) && sounding { sample(c).clamp(-1.0, 1.0) } else { 0.0 };
                }
            }
            self.read += step;
        }
    }
}

/// UI-thread bookkeeping of the third-party instances the audio thread holds.
struct Side {
    /// What was last created for each slot: (plugin id, channels, created ok).
    shadow: HashMap<(TrackId, usize), (String, usize, bool)>,
    /// Session sample rate the instances were prepared for.
    rate: f32,
    /// Editor handles of the live instances.
    editors: HashMap<(TrackId, usize), Box<dyn PluginEditor>>,
    /// State readers of the live instances.
    states: BTreeMap<(TrackId, usize), Box<dyn PluginStateReader>>,
    /// Instances that refused their stored state: saves keep that state until the plugin's editor
    /// is opened.
    refused: HashSet<(TrackId, usize)>,
    garbage: Receiver<Vec<Box<dyn Plugin>>>,
    old_sessions: Option<Receiver<Arc<Session>>>,
    /// The sound being auditioned (the audio thread has a copy of the `Arc`).
    preview: Option<Arc<PreviewSound>>,
    /// The takes recording inside the mix, while they record.
    taps: Option<record::TapWriter>,
    /// The open MIDI inputs.
    #[cfg(not(target_arch = "wasm32"))]
    midi: Option<midi_in::MidiInputs>,
}

impl Side {
    fn new(rate: f32, garbage: Receiver<Vec<Box<dyn Plugin>>>) -> Side {
        Side {
            shadow: HashMap::new(),
            rate,
            editors: HashMap::new(),
            states: BTreeMap::new(),
            refused: HashSet::new(),
            garbage,
            old_sessions: None,
            taps: None,
            preview: None,
            #[cfg(not(target_arch = "wasm32"))]
            midi: None,
        }
    }

    /// Drops retired instances (closing their editors first, inside the plugin's drop) and
    /// replaced session snapshots.
    fn collect_garbage(&mut self) {
        while let Ok(g) = self.garbage.try_recv() {
            for p in g {
                log::debug!("plugin instance released: {}", p.info().name);
                drop(p);
            }
        }
        if let Some(rx) = &self.old_sessions {
            rx.try_iter().for_each(drop);
        }
    }

    fn return_old_sessions(&mut self, tx: &Sender<Cmd>) {
        let (otx, orx) = sync_channel(OLD_SESSIONS);
        self.old_sessions = Some(orx);
        let _ = tx.send(Cmd::OldSessions(otx));
    }

    fn forget(&mut self, key: (TrackId, usize)) {
        self.shadow.remove(&key);
        self.states.remove(&key);
        self.refused.remove(&key);
        if let Some(mut e) = self.editors.remove(&key) {
            e.close();
        }
    }
}

/// The transport and audio device.
pub struct Player {
    shared: Arc<Shared>,
    fault: Arc<Fault>,
    /// The block count last seen while playing, and since when.
    #[cfg(not(target_arch = "wasm32"))]
    watch: Mutex<Option<(u64, std::time::Instant)>>,
    tx: Sender<Cmd>,
    side: Mutex<Side>,
    #[cfg(not(target_arch = "wasm32"))]
    _stream: Option<cpal::Stream>,
    #[cfg(target_arch = "wasm32")]
    _stream: Option<cpal::Stream>,
    pub device_name: String,
    pub device_rate: u32,
    /// Output channels the device was opened with (0 for the silent clock).
    pub device_channels: u16,
    /// The device buffer in frames when one was chosen (`None`: the device default).
    pub device_buffer: Option<u32>,
    /// True when no audio device could be opened (silent clock).
    pub silent: bool,
}

impl Player {
    /// Open the default output device. Never fails: without a device it runs a silent clock.
    pub fn new(session: Arc<Session>) -> Player {
        Player::with_config(session, &AudioConfig::default())
    }

    /// Open the output device `cfg` chooses (system, device, rate, buffer size), falling back to the
    /// defaults for whatever is unavailable. Never fails: without a device it runs a silent clock.
    pub fn with_config(session: Arc<Session>, cfg: &AudioConfig) -> Player {
        let p = Player::open(Arc::clone(&session), cfg);
        p.side().return_old_sessions(&p.tx);
        // Hand the initial session's third-party plugins over.
        p.update_session(session);
        p
    }

    /// What stopped the audio output, once: a bug in the engine, or the device went away or
    /// stopped asking for audio while playing.
    pub fn trouble(&self) -> Option<OutputTrouble> {
        #[cfg(not(target_arch = "wasm32"))]
        let stalled = {
            let playing = self.shared.playing.load(Ordering::Relaxed) && !self.silent;
            stalled(&mut self.watch.lock().unwrap_or_else(PoisonError::into_inner), self.fault.beats.load(Ordering::Relaxed), playing)
        };
        #[cfg(target_arch = "wasm32")]
        let stalled = false;
        self.fault.trouble(stalled)
    }

    fn side(&self) -> std::sync::MutexGuard<'_, Side> {
        self.side.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn open(session: Arc<Session>, cfg: &AudioConfig) -> Player {
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            meters: Mutex::new(MeterSnapshot::default()),
            launch: Mutex::new(LaunchStatus::reserved()),
            played: Mutex::new(Vec::with_capacity(PLAYED_CAPACITY)),
            previewing: AtomicBool::new(false),
        });
        let want = session.sample_rate.hz();
        let want_ch = u16::try_from(main_channels(&session)).unwrap_or(2);
        let side_rate = session.sample_rate.as_f64() as f32;
        let new_side = |garbage: Receiver<Vec<Box<dyn Plugin>>>| Mutex::new(Side::new(side_rate, garbage));
        match open_device(want, want_ch, cfg) {
            Ok((device, config, format, name)) => {
                // The chosen buffer size first; if the device refuses it, its default.
                let mut attempts = vec![config];
                if config.buffer_size != cpal::BufferSize::Default {
                    attempts.push(cpal::StreamConfig { buffer_size: cpal::BufferSize::Default, ..config });
                }
                let opened = first_started(
                    attempts,
                    |config: &cpal::StreamConfig| -> Result<Opened, cpal::Error> {
                        let rate = config.sample_rate;
                        let channels = usize::from(config.channels);
                        let (tx, rx) = channel();
                        let (gtx, grx) = channel();
                        let state = AudioState::new(rx, Arc::clone(&shared), Arc::clone(&session), f64::from(rate), gtx);
                        let fault = Arc::new(Fault::default());
                        let stream = match format {
                            cpal::SampleFormat::I32 => build_output::<i32>(&device, config, state, channels, &fault, &shared),
                            cpal::SampleFormat::I16 => build_output::<i16>(&device, config, state, channels, &fault, &shared),
                            _ => build_output::<f32>(&device, config, state, channels, &fault, &shared),
                        }?;
                        Ok(Opened { stream, tx, grx, fault })
                    },
                    |opened| start_stream(&opened.stream),
                );
                match opened {
                    Some((config, Opened { stream, tx, grx, fault })) => {
                        let rate = config.sample_rate;
                        let channels = usize::from(config.channels);
                        let buffer = match config.buffer_size {
                            cpal::BufferSize::Fixed(n) => Some(n),
                            cpal::BufferSize::Default => None,
                        };
                        log::info!(
                            "audio device: {name} @ {rate} Hz, {channels} ch, {format}, buffer {}",
                            buffer.map_or("default".to_string(), |n| n.to_string())
                        );
                        Player {
                            shared,
                            fault,
                            #[cfg(not(target_arch = "wasm32"))]
                            watch: Mutex::new(None),
                            tx,
                            side: new_side(grx),
                            _stream: Some(stream),
                            device_name: name,
                            device_rate: rate,
                            device_channels: config.channels,
                            device_buffer: buffer,
                            silent: false,
                        }
                    }
                    None => {
                        log::warn!("no usable audio stream; using a silent clock");
                        let (tx, rx) = channel();
                        let (gtx, grx) = channel();
                        Player::silent_clock(shared, tx, rx, want, new_side(grx), gtx)
                    }
                }
            }
            Err(e) => {
                log::warn!("no audio output: {e}; using a silent clock");
                let (tx, rx) = channel();
                let (gtx, grx) = channel();
                Player::silent_clock_with(shared, tx, rx, Arc::clone(&session), new_side(grx), gtx)
            }
        }
    }

    fn silent_clock(
        shared: Arc<Shared>,
        tx: Sender<Cmd>,
        rx: Receiver<Cmd>,
        rate: u32,
        side: Mutex<Side>,
        gtx: Sender<Vec<Box<dyn Plugin>>>,
    ) -> Player {
        let s = Arc::new(Session::new("Untitled", soundcraft_time::SampleRate::new(rate).unwrap_or_default()));
        Player::silent_clock_with(shared, tx, rx, s, side, gtx)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn silent_clock_with(
        shared: Arc<Shared>,
        tx: Sender<Cmd>,
        rx: Receiver<Cmd>,
        session: Arc<Session>,
        side: Mutex<Side>,
        gtx: Sender<Vec<Box<dyn Plugin>>>,
    ) -> Player {
        let rate = session.sample_rate.hz();
        let mut state = AudioState::new(rx, Arc::clone(&shared), session, f64::from(rate), gtx);
        let fault = Arc::new(Fault::default());
        let failed = Arc::clone(&fault);
        let spawn = std::thread::Builder::new().name("soundcraft-clock".into()).spawn(move || {
            let mut buf = vec![0.0f32; 4096];
            let mut last = std::time::Instant::now();
            loop {
                std::thread::sleep(std::time::Duration::from_millis(10));
                let now = std::time::Instant::now();
                let frames = ((now - last).as_secs_f64() * state.device_rate) as usize;
                last = now;
                let n = (frames * 2).min(buf.len());
                if let Some(b) = buf.get_mut(..n) {
                    let shared = Arc::clone(&state.shared);
                    guarded_block(&failed, &shared, b, |b| state.fill(b, 2));
                }
                if Arc::strong_count(&state.shared) <= 1 {
                    break;
                }
            }
        });
        if let Err(e) = spawn {
            log::warn!("clock thread failed: {e}");
        }
        Player {
            shared,
            fault,
            watch: Mutex::new(None),
            tx,
            side,
            _stream: None,
            device_name: "No audio device (silent)".into(),
            device_rate: rate,
            device_channels: 0,
            device_buffer: None,
            silent: true,
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn silent_clock_with(
        shared: Arc<Shared>,
        tx: Sender<Cmd>,
        _rx: Receiver<Cmd>,
        session: Arc<Session>,
        side: Mutex<Side>,
        _gtx: Sender<Vec<Box<dyn Plugin>>>,
    ) -> Player {
        Player {
            shared,
            fault: Arc::default(),
            tx,
            side,
            _stream: None,
            device_name: "No audio device".into(),
            device_rate: session.sample_rate.hz(),
            device_channels: 0,
            device_buffer: None,
            silent: true,
        }
    }

    /// Sends a new session snapshot to the audio thread, with freshly created (state-restored,
    /// prepared) instances of any third-party plugin it newly needs. Call on the UI thread.
    pub fn update_session(&self, s: Arc<Session>) {
        let instances = self.side().prepare(&s);
        let _ = self.tx.send(Cmd::Session(s, instances));
    }

    /// Reads the state of every live third-party plugin, for saving the session.
    pub fn capture_states(&self) -> PluginStates {
        self.side().capture_states()
    }

    /// Per-UI-frame housekeeping: drops retired instances and services plugin editors. Returns
    /// the parameter edits made in open editors since the last call.
    pub fn idle(&self) -> Vec<EditorEdit> {
        let mut side = self.side();
        side.collect_garbage();
        let mut out = Vec::new();
        for (&(t, slot), e) in side.editors.iter_mut() {
            for (id, v) in e.idle() {
                out.push((t, slot, id, v));
            }
        }
        out
    }

    /// True when the plugin in this slot has its own editor.
    pub fn has_editor(&self, track: TrackId, slot: usize) -> bool {
        self.side().editors.contains_key(&(track, slot))
    }

    /// Opens the plugin's own editor window. Main thread only.
    pub fn open_editor(&self, track: TrackId, slot: usize) -> Result<(), String> {
        self.side().open_editor((track, slot))
    }

    pub fn close_editor(&self, track: TrackId, slot: usize) {
        if let Some(e) = self.side().editors.get_mut(&(track, slot)) {
            e.close();
        }
    }

    /// Closes every plugin editor window (main thread). Call before replacing the player: its
    /// instances may be dropped on the audio thread, where editor windows cannot be closed.
    pub fn close_editors(&self) {
        for e in self.side().editors.values_mut() {
            e.close();
        }
    }

    pub fn editor_open(&self, track: TrackId, slot: usize) -> bool {
        self.side().editors.get(&(track, slot)).is_some_and(|e| e.is_open())
    }

    pub fn any_editor_open(&self) -> bool {
        self.side().editors.values().any(|e| e.is_open())
    }

    /// Shows a parameter change made elsewhere in the plugin's open editor.
    pub fn editor_set_param(&self, track: TrackId, slot: usize, id: &str, value: f32) {
        if let Some(e) = self.side().editors.get_mut(&(track, slot)) {
            e.set_param(id, value);
        }
    }

    /// The plugin's own text for a parameter value when SoundCraft's format would be wrong for it
    /// (see `PluginEditor::value_text`). Main thread only.
    pub fn editor_value_text(&self, track: TrackId, slot: usize, id: &str, value: f32) -> Option<String> {
        self.side().editors.get_mut(&(track, slot))?.value_text(id, value)
    }

    /// Start playback at `from`, stopping at `end`, or looping `looped`.
    pub fn play(&self, from: Samples, end: Option<Samples>, looped: Option<Range>) {
        self.shared.position.store(from, Ordering::Relaxed);
        self.shared.playing.store(true, Ordering::Relaxed);
        let _ = self.tx.send(Cmd::Play { from, end, looped });
    }

    /// Feed live input (from a `record::Recorder`) into the mix for input monitoring.
    pub fn set_input(&self, ring: Arc<record::InputRing>) {
        let _ = self.tx.send(Cmd::Input(ring));
    }

    /// Starts recording takes inside the mix: each `(track, source, channels, file)` records its
    /// source (another track's output after its inserts, or the main mix) at the session's
    /// `rate`, from the next block on while the transport records. Stop with [`Player::finish_taps`].
    pub fn record_taps(&self, takes: Vec<(TrackId, record::TapSource, usize, std::path::PathBuf)>, rate: u32) -> Result<(), String> {
        let mut side = self.side();
        if side.taps.is_some() {
            return Err("already recording".into());
        }
        let (feed, writer) = record::start_taps(takes, rate)?;
        side.taps = Some(writer);
        let _ = self.tx.send(Cmd::Taps(Some(feed)));
        Ok(())
    }

    /// Stops the takes recording inside the mix: the ones that hold audio, and the first problem.
    pub fn finish_taps(&self) -> record::Takes {
        let _ = self.tx.send(Cmd::Taps(None));
        let writer = self.side().taps.take();
        writer.map(record::TapWriter::finish).unwrap_or_default()
    }

    /// Opens every MIDI input (again: devices plugged in since are found), returning their names.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open_midi_inputs(&self) -> Vec<String> {
        let mut side = self.side();
        side.midi = None;
        let inputs = midi_in::open(&self.tx);
        let names = inputs.names.clone();
        side.midi = Some(inputs);
        let _ = self.tx.send(Cmd::MidiDevices(Arc::new(names.clone())));
        names
    }

    /// The names of the open MIDI inputs.
    pub fn midi_inputs(&self) -> Vec<String> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.side().midi.as_ref().map(|m| m.names.clone()).unwrap_or_default()
        }
        #[cfg(target_arch = "wasm32")]
        Vec::new()
    }

    /// Auditions `sound` once over the mix, also while stopped (`None` stops it).
    pub fn preview(&self, sound: Option<PreviewSound>) {
        let sound = sound.map(Arc::new);
        // Kept here as well, so the audio thread never frees it.
        self.side().preview = sound.clone();
        self.shared.previewing.store(sound.is_some(), Ordering::Relaxed);
        let _ = self.tx.send(Cmd::Preview(sound));
    }

    /// Whether a sound is being auditioned.
    pub fn previewing(&self) -> bool {
        self.shared.previewing.load(Ordering::Relaxed)
    }

    /// Plays a MIDI message as if it came from `device` ([`KEYBOARD`] for the computer keyboard).
    pub fn play_midi(&self, device: u16, data: [u8; 3]) {
        let _ = self.tx.send(Cmd::Midi { device, data });
    }

    /// Plays a note on or off straight into `track` (on-screen pads and keyboards), whatever MIDI
    /// device it listens to. The track must play live MIDI: armed or monitoring.
    pub fn play_note(&self, track: TrackId, data: [u8; 3]) {
        let _ = self.tx.send(Cmd::TrackMidi { track, data });
    }

    /// The live notes played into record-enabled tracks while recording since the last call.
    pub fn take_played(&self) -> Vec<PlayedNote> {
        self.shared.played.lock().map(|mut p| p.drain(..).collect()).unwrap_or_default()
    }

    pub fn set_recording(&self, on: bool) {
        let _ = self.tx.send(Cmd::Recording(on));
    }

    pub fn stop(&self) {
        self.shared.playing.store(false, Ordering::Relaxed);
        let _ = self.tx.send(Cmd::Stop);
    }

    pub fn locate(&self, at: Samples) {
        self.shared.position.store(at, Ordering::Relaxed);
        let _ = self.tx.send(Cmd::Locate(at));
    }

    /// Launches `scene` on `track` (stops it with `None`) at the next multiple of `quantize_bars`
    /// bars, sample-accurately, and switches the tracks to their launched slots (session mode).
    /// Start the transport first if it is stopped: the launch then begins at once.
    /// With `legato`, a slot replacing one that plays continues from that slot's loop position.
    pub fn launch(&self, track: TrackId, scene: Option<usize>, quantize_bars: f64, legato: bool) {
        let quantize_bars = if quantize_bars.is_finite() { quantize_bars.clamp(0.0, 16.0) } else { 1.0 };
        let _ = self.tx.send(Cmd::Launch { track, scene, quantize_bars, legato });
    }

    /// Plays `scene` on `track` straight away, as if launched at `start`: a loop picks up where
    /// it was (after a transport restart, say) instead of starting over.
    pub fn resume(&self, track: TrackId, scene: usize, start: Samples) {
        let _ = self.tx.send(Cmd::Resume { track, scene, start });
    }

    /// Launches `scene` on `track` exactly where the slot it plays ends its current loop pass
    /// (pattern chains, "switch at the end of the loop"); at the next multiple of `quantize_bars`
    /// when it plays nothing.
    pub fn launch_next(&self, track: TrackId, scene: usize, quantize_bars: f64) {
        let quantize_bars = if quantize_bars.is_finite() { quantize_bars.clamp(0.0, 16.0) } else { 1.0 };
        let _ = self.tx.send(Cmd::LaunchNext { track, scene, quantize_bars });
    }

    /// Tracks play their launched slots (on) or the timeline (off, which stops every slot).
    pub fn set_session_mode(&self, on: bool) {
        let _ = self.tx.send(Cmd::SessionMode(on));
    }

    /// What the launcher is playing and has queued.
    pub fn launch_status(&self) -> LaunchStatus {
        self.shared.launch.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    pub fn set_speed(&self, speed: f32) {
        let v = if speed.is_finite() { (speed.clamp(0.1, 4.0) * 1000.0) as u32 } else { 1000 };
        self.shared.speed.store(v, Ordering::Relaxed);
    }

    /// Playback rate last requested, 1.0 at full speed.
    pub fn speed(&self) -> f32 {
        self.shared.speed.load(Ordering::Relaxed) as f32 / 1000.0
    }

    pub fn position(&self) -> Samples {
        self.shared.position.load(Ordering::Relaxed)
    }

    pub fn is_playing(&self) -> bool {
        self.shared.playing.load(Ordering::Relaxed)
    }

    pub fn meters(&self) -> MeterSnapshot {
        self.shared.meters.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    /// Copy meters into reusable UI storage. Once track/channel counts are stable this performs
    /// no allocation, unlike cloning a new snapshot every video frame.
    pub fn copy_meters(&self, out: &mut MeterSnapshot) {
        let source = self.shared.meters.lock().unwrap_or_else(PoisonError::into_inner);
        copy_meter_snapshot(&source, out);
    }

    /// The main mix's live loudness (see [`Loudness`]).
    pub fn loudness(&self) -> Loudness {
        self.shared.meters.lock().unwrap_or_else(PoisonError::into_inner).loudness
    }
}

fn copy_meter_snapshot(source: &MeterSnapshot, out: &mut MeterSnapshot) {
    out.tracks.retain(|id, _| source.tracks.contains_key(id));
    for (id, meter) in &source.tracks {
        out.tracks.entry(*id).or_default().copy_from(meter);
    }
    out.main.copy_from(&source.main);
    out.loudness = source.loudness;
}

impl Side {
    fn capture_states(&mut self) -> PluginStates {
        self.states.iter_mut().filter(|(k, _)| !self.refused.contains(*k)).filter_map(|(&(t, slot), r)| Some((t, slot, r.save_state()?))).collect()
    }

    fn open_editor(&mut self, key: (TrackId, usize)) -> Result<(), String> {
        let e = self.editors.get_mut(&key).ok_or_else(|| "this plugin has no editor (or it is not loaded)".to_string())?;
        e.open()?;
        self.refused.remove(&key);
        Ok(())
    }

    /// Creates the third-party instances `s` needs that the audio thread does not have yet.
    fn prepare(&mut self, s: &Session) -> Vec<PreparedInstance> {
        self.collect_garbage();
        let sr = s.sample_rate.as_f64() as f32;
        if (sr - self.rate).abs() > 0.5 {
            // The audio thread rebuilds its engine: everything is created anew.
            let keys: Vec<(TrackId, usize)> = self.shadow.keys().copied().collect();
            for k in keys {
                self.forget(k);
            }
            self.rate = sr;
        }
        let specs = soundcraft_mix::instance_specs(s);
        let stale: Vec<(TrackId, usize)> = self.shadow.keys().filter(|k| !specs.iter().any(|x| (x.track, x.slot) == **k)).copied().collect();
        for k in stale {
            self.forget(k);
        }
        let mut out = Vec::new();
        for spec in specs {
            let key = (spec.track, spec.slot);
            if self.shadow.get(&key).is_some_and(|(id, ch, _)| *id == spec.id && *ch == spec.ch) {
                continue;
            }
            self.forget(key);
            let made = soundcraft_mix::create_instance(&spec.id, spec.state.as_deref(), sr, BLOCK, spec.ch);
            log::debug!("plugin instance for {:?} slot {}: {} ({} ch, created: {})", spec.track, spec.slot, spec.id, spec.ch, made.is_some());
            self.shadow.insert(key, (spec.id.clone(), spec.ch, made.is_some()));
            if let Some(soundcraft_mix::NewInstance { mut plugin, refused_state }) = made {
                if let Some(e) = plugin.editor() {
                    self.editors.insert(key, e);
                }
                if let Some(r) = plugin.state_reader() {
                    self.states.insert(key, r);
                }
                if refused_state {
                    self.refused.insert(key);
                }
                out.push(PreparedInstance { track: spec.track, slot: spec.slot, id: spec.id, ch: spec.ch, plugin });
            }
        }
        out
    }
}

/// ITU stereo fold-down gains for each main channel of a session.
fn fold_gains(s: &Session) -> [(f32, f32); MAX_CHANNELS] {
    let mut g = [(0.0, 0.0); MAX_CHANNELS];
    let f = s.main_format();
    let sp = f.speakers();
    if sp.len() == f.channels() {
        for (o, x) in g.iter_mut().zip(sp.iter()) {
            *o = itu_stereo_fold(*x);
        }
    } else {
        // Ambisonics: W only, at -3 dB per side.
        g[0] = (std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2);
    }
    g
}

/// Start a freshly built output stream. `false` when the device refuses to start it, so the caller
/// treats it like a stream that failed to build and tries the next configuration.
fn start_stream(stream: &impl cpal::traits::StreamTrait) -> bool {
    match stream.play() {
        Ok(()) => true,
        Err(e) => {
            log::warn!("audio stream failed to start: {e}");
            false
        }
    }
}

/// A built output stream and the channels and fault flag its callback shares with the player.
struct Opened {
    stream: cpal::Stream,
    tx: Sender<Cmd>,
    grx: Receiver<Vec<Box<dyn Plugin>>>,
    fault: Arc<Fault>,
}

/// Build and start the attempts in order; the first that builds and starts wins. A build error or
/// a stream that will not start is logged, and the next attempt is tried. `None` when none works.
fn first_started<A: std::fmt::Debug, S, E: std::fmt::Display>(
    attempts: impl IntoIterator<Item = A>,
    mut build: impl FnMut(&A) -> Result<S, E>,
    start: impl Fn(&S) -> bool,
) -> Option<(A, S)> {
    for attempt in attempts {
        match build(&attempt) {
            Ok(built) if start(&built) => return Some((attempt, built)),
            // `start` has already logged why the stream would not start.
            Ok(_) => {}
            Err(e) => log::warn!("cannot open audio stream ({attempt:?}): {e}"),
        }
    }
    None
}

/// The output stream: the mix renders f32 into a scratch buffer allocated here (so the callback
/// never allocates), converted to the device's sample type.
fn build_output<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut state: AudioState,
    channels: usize,
    fault: &Arc<Fault>,
    shared: &Arc<Shared>,
) -> Result<cpal::Stream, cpal::Error>
where
    T: cpal::SizedSample + cpal::FromSample<f32>,
{
    use cpal::traits::DeviceTrait;
    let frame = channels.max(1);
    let mut scratch = vec![0.0f32; 8192 * frame];
    let chunk = scratch.len();
    let (failed, guard, gone) = (Arc::clone(fault), Arc::clone(shared), Arc::clone(fault));
    device.build_output_stream(
        *config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            mark_audio_thread();
            for out in data.chunks_mut(chunk) {
                let Some(s) = scratch.get_mut(..out.len()) else { return };
                guarded_block(&failed, &guard, s, |d| state.fill(d, channels));
                for (o, x) in out.iter_mut().zip(s.iter()) {
                    *o = T::from_sample(*x);
                }
            }
        },
        // cpal may call this on the audio thread (on ALSA in a loop while the device stays
        // broken): marked, so the logger hands the record off instead of writing it.
        move |e: cpal::Error| {
            mark_audio_thread();
            if e.kind() == cpal::ErrorKind::DeviceNotAvailable {
                gone.gone.store(true, Ordering::Relaxed);
            }
            log::warn!("audio stream error: {e}");
        },
        None,
    )
}

/// Open the output device `cfg` chooses (else the default) at the chosen rate (else the session
/// rate) if possible, in the best sample format it offers (f32, else 32- or 16-bit integers),
/// with as many channels as it offers up to `want_ch` (at least stereo when available), and the
/// chosen buffer size when the device allows it.
fn open_device(want_rate: u32, want_ch: u16, cfg: &AudioConfig) -> Result<(cpal::Device, cpal::StreamConfig, cpal::SampleFormat, String), String> {
    use cpal::traits::DeviceTrait;
    let host = devices::host(cfg);
    let device = devices::device(&host, cfg.output.as_deref(), true).ok_or_else(|| "no output device".to_string())?;
    let name = devices::device_name(&device).unwrap_or_else(|| "Audio device".into());
    // Prefer a config at the chosen (else session) rate; else the device default (we resample).
    let want_ch = want_ch.max(2);
    let rates: Vec<u32> = cfg.sample_rate.into_iter().chain(Some(want_rate)).collect();
    let mut chosen: Option<(cpal::StreamConfig, cpal::SupportedBufferSize, cpal::SampleFormat)> = None;
    // Better: a preferred format first, then more channels.
    let better = |c: &cpal::SupportedStreamConfigRange, x: &(cpal::StreamConfig, cpal::SupportedBufferSize, cpal::SampleFormat)| {
        let (rc, rx) = (devices::format_rank(c.sample_format()).unwrap_or(usize::MAX), devices::format_rank(x.2).unwrap_or(usize::MAX));
        rc < rx || (rc == rx && c.channels() > x.0.channels)
    };
    if let Ok(configs) = device.supported_output_configs() {
        let configs: Vec<cpal::SupportedStreamConfigRange> = configs.collect();
        for rate in rates {
            for c in &configs {
                if devices::format_rank(c.sample_format()).is_some()
                    && c.min_sample_rate() <= rate
                    && c.max_sample_rate() >= rate
                    && c.channels() >= 2
                    && c.channels() <= want_ch
                    && chosen.as_ref().is_none_or(|x| better(c, x))
                {
                    chosen = Some((c.with_sample_rate(rate).config(), *c.buffer_size(), c.sample_format()));
                }
            }
            if chosen.is_some() {
                break;
            }
        }
    }
    let (config, format) = match chosen {
        Some((mut c, buffers, format)) => {
            c.buffer_size = devices::buffer_size(cfg.buffer_size, &buffers);
            (c, format)
        }
        None => {
            let d = device.default_output_config().map_err(|e| e.to_string())?;
            if devices::format_rank(d.sample_format()).is_none() {
                return Err(format!("unsupported device sample format {:?}", d.sample_format()));
            }
            let buffers = *d.buffer_size();
            let format = d.sample_format();
            let mut c = d.config();
            c.buffer_size = devices::buffer_size(cfg.buffer_size, &buffers);
            (c, format)
        }
    };
    Ok((device, config, format, name))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stream that builds; `starts` says whether the device lets it play (it can refuse after a
    /// reconnect).
    struct Fake {
        starts: bool,
    }

    impl cpal::traits::StreamTrait for Fake {
        /// Play reports the device's refusal when `starts` is false.
        fn play(&self) -> Result<(), cpal::Error> {
            if self.starts { Ok(()) } else { Err(cpal::Error::new(cpal::ErrorKind::DeviceNotAvailable)) }
        }
        /// Pause always succeeds on the fake stream.
        fn pause(&self) -> Result<(), cpal::Error> {
            Ok(())
        }
        /// The fake stream claims one fixed buffer.
        fn buffer_size(&self) -> Result<cpal::FrameCount, cpal::Error> {
            Ok(1024)
        }
        /// The fake stream has no clock.
        fn now(&self) -> cpal::StreamInstant {
            cpal::StreamInstant::new(0, 0)
        }
    }

    #[test]
    fn a_stream_that_will_not_start_is_reported_as_failed() {
        assert!(!start_stream(&Fake { starts: false }));
        assert!(start_stream(&Fake { starts: true }));
    }

    #[test]
    fn the_first_config_that_starts_wins_and_later_ones_are_not_tried() {
        let mut tried = Vec::new();
        let won = first_started(
            [1u32, 2, 3],
            |&n| {
                tried.push(n);
                Ok::<_, String>(Fake { starts: n != 1 })
            },
            start_stream,
        );
        assert_eq!(won.map(|(n, _)| n), Some(2));
        assert_eq!(tried, vec![1, 2]);
    }

    #[test]
    fn a_config_that_fails_to_build_falls_through_to_the_next() {
        let won =
            first_started([1u32, 2], |&n| if n == 1 { Err("no such buffer size".to_string()) } else { Ok(Fake { starts: true }) }, start_stream);
        assert_eq!(won.map(|(n, _)| n), Some(2));
    }

    #[test]
    fn no_config_that_starts_yields_none_so_the_caller_falls_back_to_a_silent_clock() {
        let mut tried = 0;
        let won = first_started(
            [1u32, 2],
            |_| {
                tried += 1;
                Ok::<_, String>(Fake { starts: false })
            },
            start_stream,
        );
        assert!(won.is_none());
        assert_eq!(tried, 2);
    }

    #[test]
    fn reusable_meter_snapshot_updates_and_drops_stale_tracks() {
        let keep = TrackId(1);
        let stale = TrackId(2);
        let mut source = MeterSnapshot::default();
        source.tracks.insert(keep, StripMeter { peak: [0.5, 0.25], peaks: vec![0.5, 0.25], gain_reduction_db: 3.0, ..StripMeter::default() });
        source.main.peaks = vec![0.75, 0.5];
        let mut out = MeterSnapshot::default();
        out.tracks.insert(stale, StripMeter::default());
        copy_meter_snapshot(&source, &mut out);
        assert_eq!(out.tracks.len(), 1);
        assert_eq!(out.tracks[&keep], source.tracks[&keep]);
        assert_eq!(out.main, source.main);
    }

    #[test]
    fn a_block_that_panics_goes_silent_and_is_reported_once() {
        let shared = Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(true),
            speed: AtomicU32::new(1000),
            meters: Mutex::new(MeterSnapshot::default()),
            launch: Mutex::new(LaunchStatus::reserved()),
            played: Mutex::new(Vec::with_capacity(PLAYED_CAPACITY)),
            previewing: AtomicBool::new(false),
        };
        let fault = Fault::default();
        let mut out = [0.5f32; 8];
        guarded_block(&fault, &shared, &mut out, |o| o.fill(0.25));
        assert_eq!((out, fault.trouble(false)), ([0.25; 8], None));
        guarded_block(&fault, &shared, &mut out, |_| std::panic::panic_any("a bug in the mix"));
        guarded_block(&fault, &shared, &mut out, |o| o.fill(0.25));
        assert_eq!(out, [0.0; 8]);
        assert!(!shared.playing.load(Ordering::Relaxed));
        assert_eq!((fault.trouble(false), fault.trouble(false)), (Some(OutputTrouble::Bug), None));
        assert_eq!(fault.beats.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn a_lost_or_stalled_device_is_reported_once() {
        let ago = |seen: &mut Option<(u64, std::time::Instant)>| {
            if let Some(s) = seen {
                s.1 = std::time::Instant::now().checked_sub(STALL).unwrap();
            }
        };
        let mut seen = None;
        assert!(!stalled(&mut seen, 5, true));
        ago(&mut seen);
        assert!(stalled(&mut seen, 5, true));
        assert!(!stalled(&mut seen, 6, true), "it asked again");
        ago(&mut seen);
        assert!(!stalled(&mut seen, 6, false), "stopped, so nothing to ask for");
        assert_eq!(Fault::default().trouble(true), Some(OutputTrouble::Stalled));
        let fault = Fault::default();
        fault.gone.store(true, Ordering::Relaxed);
        assert_eq!((fault.trouble(false), fault.trouble(true)), (Some(OutputTrouble::Gone), None));
    }

    #[test]
    fn audio_state_plays_and_stops_at_end() {
        let s = Arc::new(soundcraft_model::Session::default());
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            meters: Mutex::new(MeterSnapshot::default()),
            launch: Mutex::new(LaunchStatus::reserved()),
            played: Mutex::new(Vec::with_capacity(PLAYED_CAPACITY)),
            previewing: AtomicBool::new(false),
        });
        let (tx, rx) = channel();
        let mut st = AudioState::new(rx, Arc::clone(&shared), s, 48_000.0, channel().0);
        tx.send(Cmd::Play { from: 0, end: Some(1000), looped: None }).unwrap();
        let mut out = vec![0.0f32; 4096];
        st.fill(&mut out, 2);
        assert!(shared.position.load(Ordering::Relaxed) >= 1000);
        assert!(!shared.playing.load(Ordering::Relaxed));
    }

    #[test]
    fn short_device_callbacks_render_short_mix_blocks() {
        let s = Arc::new(soundcraft_model::Session::default());
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            meters: Mutex::new(MeterSnapshot::default()),
            launch: Mutex::new(LaunchStatus::reserved()),
            played: Mutex::new(Vec::with_capacity(PLAYED_CAPACITY)),
            previewing: AtomicBool::new(false),
        });
        let (tx, rx) = channel();
        let mut st = AudioState::new(rx, Arc::clone(&shared), s, 48_000.0, channel().0);
        tx.send(Cmd::Play { from: 0, end: None, looped: None }).unwrap();
        let mut out = vec![0.0f32; 2 * 64];
        st.fill(&mut out, 2);
        assert_eq!(shared.position.load(Ordering::Relaxed), 64);
    }

    fn audio_output_state(format: soundcraft_model::ChannelFormat, channels: Vec<Vec<f32>>) -> AudioState {
        use soundcraft_model::{Clip, SourceAudio, SourceId, TrackKind};
        let mut s = Session::default();
        s.edit.click = false;
        s.outputs[0].format = format;
        let t = s.add_track(TrackKind::Audio, format, None);
        let frames = channels[0].len();
        s.pool.insert(SourceId(5), Arc::new(SourceAudio::new(soundcraft_audio_io::AudioBuffer { sample_rate: 48_000, channels })));
        let id = s.new_clip_id();
        s.track_mut(t).unwrap().playlist_mut().unwrap().clips.push(Clip::audio(id, "output", SourceId(5), 0, 0, frames as i64));
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            meters: Mutex::new(MeterSnapshot::default()),
            launch: Mutex::new(LaunchStatus::reserved()),
            played: Mutex::new(Vec::with_capacity(PLAYED_CAPACITY)),
            previewing: AtomicBool::new(false),
        });
        let (tx, rx) = channel();
        let st = AudioState::new(rx, shared, Arc::new(s), 48_000.0, channel().0);
        tx.send(Cmd::Play { from: 0, end: None, looped: None }).unwrap();
        st
    }

    #[test]
    fn a_legato_launch_continues_from_the_same_loop_position() {
        use soundcraft_model::{ChannelFormat, Clip, LaunchSlot, SourceAudio, SourceId, TrackKind};
        // 120 BPM 4/4 at 48 kHz: a beat is 24 000 samples, a bar 96 000. Slot A is a steady 0.5,
        // slot B a ramp whose value tells where in its 1-bar loop it is playing.
        let mut s = Session::default();
        s.edit.click = false;
        let t = s.add_track(TrackKind::Audio, ChannelFormat::Mono, None);
        let buf = |v: Vec<f32>| Arc::new(SourceAudio::new(soundcraft_audio_io::AudioBuffer { sample_rate: 48_000, channels: vec![v] }));
        s.pool.insert(SourceId(5), buf(vec![0.5; 96_000]));
        s.pool.insert(SourceId(6), buf((0..96_000).map(|i| i as f32 / 96_000.0).collect()));
        s.launcher.insert_scene(0, "A".into());
        s.launcher.insert_scene(1, "B".into());
        for (scene, src) in [(0, 5), (1, 6)] {
            let id = s.new_clip_id();
            s.launcher.set_slot(t, scene, Some(LaunchSlot::new(Clip::audio(id, "loop", SourceId(src), 0, 0, 96_000), 3840))).unwrap();
        }
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            meters: Mutex::new(MeterSnapshot::default()),
            launch: Mutex::new(LaunchStatus::reserved()),
            played: Mutex::new(Vec::with_capacity(PLAYED_CAPACITY)),
            previewing: AtomicBool::new(false),
        });
        let (tx, rx) = channel();
        let mut st = AudioState::new(rx, Arc::clone(&shared), Arc::new(s), 48_000.0, channel().0);
        tx.send(Cmd::Play { from: 0, end: None, looped: None }).unwrap();
        let mut out = vec![0.0f32; 2 * 500];
        let mut played = Vec::new();
        let mut run_to = |st: &mut AudioState, played: &mut Vec<f32>, n: usize| {
            while played.len() < n {
                st.fill(&mut out, 2);
                played.extend(out.iter().step_by(2).copied());
            }
        };
        // A starts on bar 1 at once; B is launched legato 1.5 beats into it, on a 1/4 quantize.
        tx.send(Cmd::Launch { track: t, scene: Some(0), quantize_bars: 0.0, legato: true }).unwrap();
        run_to(&mut st, &mut played, 36_000);
        tx.send(Cmd::Launch { track: t, scene: Some(1), quantize_bars: 0.25, legato: true }).unwrap();
        run_to(&mut st, &mut played, 60_000);
        // Levels relative to A's steady 0.5 (the track's pan law scales both alike).
        let gain = played.get(47_999).copied().unwrap_or(0.0) / 0.5;
        assert!(gain > 0.1, "A plays to the beat");
        let at = |i: usize| played.get(i).copied().unwrap_or(f32::NAN) / gain;
        assert!((at(47_000) - 0.5).abs() < 1e-4, "A is steady: {}", at(47_000));
        // On beat 3 (sample 48 000) B takes over half way through its loop, not from the top.
        assert!((at(48_000) - 0.5).abs() < 1e-3, "{}", at(48_000));
        assert!((at(59_999) - 59_999.0 / 96_000.0).abs() < 1e-3);
        let status = shared.launch.lock().unwrap().clone();
        assert_eq!(status.playing, vec![(t, 1, 0)], "B keeps A's start reference");
        // The same switch with Restart starts B from the top on the next beat.
        tx.send(Cmd::Launch { track: t, scene: Some(0), quantize_bars: 0.25, legato: false }).unwrap();
        run_to(&mut st, &mut played, 80_000);
        assert!(played.get(72_000).is_some_and(|v| (v / gain - 0.5).abs() < 1e-4));
        assert_eq!(shared.launch.lock().unwrap().playing, vec![(t, 0, 72_000)]);
    }

    #[test]
    fn launch_next_starts_where_the_playing_slot_ends_its_pass() {
        use soundcraft_model::{ChannelFormat, Clip, LaunchSlot, SourceAudio, SourceId, TrackKind};
        // 120 BPM 4/4 at 48 kHz: a bar is 96 000 samples. A is 1.5 bars, B a steady 0.25.
        let mut s = Session::default();
        s.edit.click = false;
        let t = s.add_track(TrackKind::Audio, ChannelFormat::Mono, None);
        let buf = |v: f32| Arc::new(SourceAudio::new(soundcraft_audio_io::AudioBuffer { sample_rate: 48_000, channels: vec![vec![v; 192_000]] }));
        s.pool.insert(SourceId(5), buf(0.5));
        s.pool.insert(SourceId(6), buf(0.25));
        s.launcher.insert_scene(0, "A".into());
        s.launcher.insert_scene(1, "B".into());
        for (scene, src, ticks) in [(0, 5, 5760), (1, 6, 3840)] {
            let id = s.new_clip_id();
            s.launcher.set_slot(t, scene, Some(LaunchSlot::new(Clip::audio(id, "loop", SourceId(src), 0, 0, 192_000), ticks))).unwrap();
        }
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            meters: Mutex::new(MeterSnapshot::default()),
            launch: Mutex::new(LaunchStatus::reserved()),
            played: Mutex::new(Vec::with_capacity(PLAYED_CAPACITY)),
            previewing: AtomicBool::new(false),
        });
        let (tx, rx) = channel();
        let mut st = AudioState::new(rx, Arc::clone(&shared), Arc::new(s), 48_000.0, channel().0);
        tx.send(Cmd::Play { from: 0, end: None, looped: None }).unwrap();
        tx.send(Cmd::Launch { track: t, scene: Some(0), quantize_bars: 0.0, legato: false }).unwrap();
        let mut out = vec![0.0f32; 2 * 500];
        let mut played = Vec::new();
        while played.len() < 20_000 {
            st.fill(&mut out, 2);
            played.extend(out.iter().step_by(2).copied());
        }
        // Queued a bar quantize would put at 96 000; next-pass puts it at A's end, 144 000.
        tx.send(Cmd::LaunchNext { track: t, scene: 1, quantize_bars: 1.0 }).unwrap();
        while played.len() < 150_000 {
            st.fill(&mut out, 2);
            played.extend(out.iter().step_by(2).copied());
        }
        let a = played.get(143_999).copied().unwrap_or(0.0);
        let b = played.get(144_000).copied().unwrap_or(0.0);
        assert!(a > 0.0 && (b / a - 0.5).abs() < 1e-4, "A to 143 999 ({a}), B from 144 000 ({b})");
        assert_eq!(shared.launch.lock().unwrap().playing, vec![(t, 1, 144_000)]);
    }

    #[test]
    fn a_launch_lands_exactly_on_the_next_quantise_boundary() {
        use soundcraft_model::{ChannelFormat, Clip, LaunchSlot, SourceAudio, SourceId, TrackKind};
        // 120 BPM 4/4 at 48 kHz: bar 2 starts at sample 96 000.
        let mut s = Session::default();
        s.edit.click = false;
        let t = s.add_track(TrackKind::Audio, ChannelFormat::Mono, None);
        s.pool.insert(
            SourceId(5),
            Arc::new(SourceAudio::new(soundcraft_audio_io::AudioBuffer { sample_rate: 48_000, channels: vec![vec![0.5; 96_000]] })),
        );
        s.launcher.insert_scene(0, "A".into());
        let id = s.new_clip_id();
        s.launcher.set_slot(t, 0, Some(LaunchSlot::new(Clip::audio(id, "loop", SourceId(5), 0, 0, 96_000), 3840))).unwrap();
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            meters: Mutex::new(MeterSnapshot::default()),
            launch: Mutex::new(LaunchStatus::reserved()),
            played: Mutex::new(Vec::with_capacity(PLAYED_CAPACITY)),
            previewing: AtomicBool::new(false),
        });
        let (tx, rx) = channel();
        let mut st = AudioState::new(rx, Arc::clone(&shared), Arc::new(s), 48_000.0, channel().0);
        tx.send(Cmd::Play { from: 0, end: None, looped: None }).unwrap();
        let mut out = vec![0.0f32; 2 * 500];
        let mut played = Vec::new();
        st.fill(&mut out, 2);
        played.extend(out.iter().step_by(2).copied());
        // Launched a little into bar 1: queued for bar 2.
        tx.send(Cmd::Launch { track: t, scene: Some(0), quantize_bars: 1.0, legato: false }).unwrap();
        st.fill(&mut out, 2);
        played.extend(out.iter().step_by(2).copied());
        let status = shared.launch.lock().unwrap().clone();
        assert!(status.session_mode);
        assert_eq!(status.queued, vec![(t, Some(0), 96_000)]);
        while played.len() < 100_000 {
            st.fill(&mut out, 2);
            played.extend(out.iter().step_by(2).copied());
        }
        let first = played.iter().position(|x| x.abs() > 1e-6);
        assert_eq!(first, Some(96_000), "starts on the bar line, to the sample");
        let status = shared.launch.lock().unwrap().clone();
        assert_eq!((status.playing, status.queued.len()), (vec![(t, 0, 96_000)], 0));
        // Stopping the transport stops the slot.
        tx.send(Cmd::Stop).unwrap();
        st.fill(&mut out, 2);
        assert!(shared.launch.lock().unwrap().playing.is_empty());
        // Restarting mid-loop and resuming keeps the loop's phase: it plays as launched at bar 2.
        tx.send(Cmd::Play { from: 120_000, end: None, looped: None }).unwrap();
        tx.send(Cmd::Resume { track: t, scene: 0, start: 96_000 }).unwrap();
        st.fill(&mut out, 2);
        assert_eq!(shared.launch.lock().unwrap().playing, vec![(t, 0, 96_000)]);
    }

    #[test]
    fn main_loudness_is_measured_live_and_restarts_with_playback() {
        use soundcraft_model::{ChannelFormat, Clip, SourceAudio, SourceId, TrackKind};
        let frames = 48_000 * 4;
        let tone: Vec<f32> = (0..frames).map(|i| 0.1 * (std::f32::consts::TAU * 997.0 * i as f32 / 48_000.0).sin()).collect();
        let mut s = Session::default();
        s.edit.click = false;
        let t = s.add_track(TrackKind::Audio, ChannelFormat::Stereo, None);
        s.pool.insert(
            SourceId(5),
            Arc::new(SourceAudio::new(soundcraft_audio_io::AudioBuffer { sample_rate: 48_000, channels: vec![tone.clone(), tone] })),
        );
        let id = s.new_clip_id();
        s.track_mut(t).unwrap().playlist_mut().unwrap().clips.push(Clip::audio(id, "tone", SourceId(5), 0, 0, frames as i64));
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            meters: Mutex::new(MeterSnapshot::default()),
            launch: Mutex::new(LaunchStatus::reserved()),
            played: Mutex::new(Vec::with_capacity(PLAYED_CAPACITY)),
            previewing: AtomicBool::new(false),
        });
        let (tx, rx) = channel();
        let mut st = AudioState::new(rx, Arc::clone(&shared), Arc::new(s), 48_000.0, channel().0);
        assert_eq!(shared.meters.lock().unwrap().loudness, Loudness::default(), "nothing measured before playback");
        tx.send(Cmd::Play { from: 0, end: None, looped: None }).unwrap();
        // Three seconds through a stereo device; the same output measured independently.
        let mut reference = LoudnessMeter::new(48_000.0, 2);
        let mut out = vec![0.0f32; 2 * 480];
        for _ in 0..300 {
            st.fill(&mut out, 2);
            let planar: Vec<Vec<f32>> = (0..2).map(|c| out.iter().skip(c).step_by(2).copied().collect()).collect();
            reference.process(&planar, 480);
        }
        let live = shared.meters.lock().unwrap().loudness;
        assert!(live.integrated > -40.0 && live.integrated < 0.0, "{live:?}");
        assert!((live.integrated - reference.integrated_lufs()).abs() < 0.3, "{live:?} vs {}", reference.integrated_lufs());
        assert!((live.short_term - live.integrated).abs() < 0.3, "a steady tone: {live:?}");
        assert!((live.true_peak_db - reference.true_peak_db()).abs() < 0.3, "{live:?}");
        // A stop keeps the reading; the next Play starts a new measurement.
        tx.send(Cmd::Stop).unwrap();
        st.fill(&mut out, 2);
        assert_eq!(shared.meters.lock().unwrap().loudness, live);
        tx.send(Cmd::Play { from: 0, end: None, looped: None }).unwrap();
        st.fill(&mut out, 2);
        let fresh = shared.meters.lock().unwrap().loudness;
        assert_eq!((fresh.momentary, fresh.integrated), (-144.0, -144.0), "under 400 ms measured: {fresh:?}");
    }

    #[test]
    fn mono_device_averages_stereo_before_clipping() {
        use soundcraft_model::ChannelFormat;
        // Right only, left only, identical, opposite polarity, and levels that must be averaged
        // before clipping either side to the device's range.
        let left = [0.0, 0.8, 0.8, 0.8, 2.0, -2.0, 2.0];
        let right = [0.8, 0.0, 0.8, -0.8, -0.5, 0.5, 2.0];
        let expected = [0.4, 0.4, 0.8, 0.0, 0.75, -0.75, 1.0];
        for callback in [1, 37, 512] {
            let channels = [left, right].map(|pattern| (0..2048).map(|i| pattern[i % pattern.len()]).collect());
            let mut st = audio_output_state(ChannelFormat::Stereo, channels.into());
            let mut out = vec![0.0; callback];
            for block in 0..3 {
                st.fill(&mut out, 1);
                for (i, actual) in out.iter().enumerate() {
                    let at = block * callback + i;
                    assert!((actual - expected[at % expected.len()]).abs() < 1e-6, "callback {callback}, sample {at}: {actual}");
                }
            }
        }
    }

    #[test]
    fn mono_device_preserves_centered_mono_level_and_includes_surround_right_channels() {
        use soundcraft_model::ChannelFormat;
        let h = std::f32::consts::FRAC_1_SQRT_2;
        let mut mono = audio_output_state(ChannelFormat::Mono, vec![vec![0.8; 512]]);
        let mut out = [0.0; 128];
        mono.fill(&mut out, 1);
        let centered = 0.8 * soundcraft_dsp::db_to_gain(-3.0);
        assert!(out.iter().all(|x| (*x - centered).abs() < 1e-6), "a centered mono track keeps its rendered level under the -3 dB pan law");

        // Independently exercise each SMPTE 5.1 channel: L, R, C, LFE, Ls, Rs.
        let channels = (0..6).map(|c| (0..512).map(|i| if i % 6 == c { 0.8 } else { 0.0 }).collect()).collect();
        let mut surround = audio_output_state(ChannelFormat::Surround51, channels);
        surround.fill(&mut out, 1);
        let expected = [0.4, 0.4, 0.8 * h, 0.0, 0.4 * h, 0.4 * h];
        for (i, actual) in out.iter().enumerate() {
            assert!((actual - expected[i % expected.len()]).abs() < 1e-6, "surround sample {i}: {actual}");
        }
    }

    #[test]
    fn mono_device_contains_overflow_from_opposite_surround_sides() {
        use soundcraft_model::ChannelFormat;
        let channels = [f32::MAX, -f32::MAX, 0.0, 0.0, f32::MAX, -f32::MAX].map(|v| vec![v; 128]);
        let mut st = audio_output_state(ChannelFormat::Surround51, channels.into());
        let mut out = [0.5; 128];
        st.fill(&mut out, 1);
        assert!(out.iter().all(|x| *x == 0.0), "overflow in the stereo fold-down must not send NaN to a mono device");
    }

    #[test]
    fn surround_main_maps_to_device_channels_or_folds_down() {
        use soundcraft_model::{ChannelFormat, Clip, SourceAudio, SourceId, TrackKind};
        let mut s = soundcraft_model::Session::default();
        s.outputs[0].format = ChannelFormat::Surround51;
        let t = s.add_track(TrackKind::Audio, ChannelFormat::Surround51, None);
        let chans: Vec<Vec<f32>> = (0..6).map(|c| vec![0.1 * (c + 1) as f32; 48_000]).collect();
        s.pool.insert(SourceId(5), Arc::new(SourceAudio::new(soundcraft_audio_io::AudioBuffer { sample_rate: 48_000, channels: chans })));
        let id = s.new_clip_id();
        s.track_mut(t).unwrap().playlist_mut().unwrap().clips.push(Clip::audio(id, "x", SourceId(5), 0, 0, 48_000));
        let s = Arc::new(s);
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            meters: Mutex::new(MeterSnapshot::default()),
            launch: Mutex::new(LaunchStatus::reserved()),
            played: Mutex::new(Vec::with_capacity(PLAYED_CAPACITY)),
            previewing: AtomicBool::new(false),
        });
        let (tx, rx) = channel();
        let mut st = AudioState::new(rx, Arc::clone(&shared), Arc::clone(&s), 48_000.0, channel().0);
        tx.send(Cmd::Play { from: 0, end: None, looped: None }).unwrap();
        // An 8-channel device: channels in order, the rest silent.
        let mut out = vec![0.0f32; 8 * 256];
        st.fill(&mut out, 8);
        let f = &out[8 * 100..8 * 101];
        for c in 0..6 {
            assert!((f[c] - 0.1 * (c + 1) as f32).abs() < 1e-5, "{f:?}");
        }
        assert!(f[6] == 0.0 && f[7] == 0.0);
        // A stereo device: ITU fold-down.
        let mut out = vec![0.0f32; 2 * 256];
        st.fill(&mut out, 2);
        let h = std::f32::consts::FRAC_1_SQRT_2;
        let l = (0.1 + h * 0.3 + h * 0.5f32).min(1.0);
        assert!((out[200] - l).abs() < 1e-4, "{} vs {l}", out[200]);
        assert_eq!(shared.meters.lock().unwrap().main.peaks.len(), 6);
    }

    #[test]
    fn instances_arrive_with_the_session_and_leave_through_the_return_channel() {
        use soundcraft_model::{ChannelFormat, Insert, TrackKind};
        let mut s = soundcraft_model::Session::default();
        let t = s.add_track(TrackKind::Audio, ChannelFormat::Stereo, None);
        s.track_mut(t).unwrap().mixer.inserts[1] = Some(Insert::new("clap:test.fake"));
        let s = Arc::new(s);
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            meters: Mutex::new(MeterSnapshot::default()),
            launch: Mutex::new(LaunchStatus::reserved()),
            played: Mutex::new(Vec::with_capacity(PLAYED_CAPACITY)),
            previewing: AtomicBool::new(false),
        });
        let (tx, rx) = channel();
        let (gtx, grx) = channel();
        let mut st = AudioState::new(rx, Arc::clone(&shared), Arc::new(soundcraft_model::Session::default()), 48_000.0, gtx);
        let plugin = soundcraft_dsp::create("gain").unwrap();
        tx.send(Cmd::Session(Arc::clone(&s), vec![PreparedInstance { track: t, slot: 1, id: "clap:test.fake".into(), ch: 2, plugin }])).unwrap();
        let mut out = vec![0.0f32; 256];
        st.fill(&mut out, 2);
        assert!(grx.try_recv().is_err(), "nothing retired yet");
        // A session without the insert retires the instance to the return channel.
        tx.send(Cmd::Session(Arc::new(soundcraft_model::Session::default()), Vec::new())).unwrap();
        st.fill(&mut out, 2);
        assert_eq!(grx.try_recv().unwrap().len(), 1);
        // A sample-rate change rebuilds the engine without dropping plugins on this thread.
        tx.send(Cmd::Session(
            Arc::clone(&s),
            vec![PreparedInstance { track: t, slot: 1, id: "clap:test.fake".into(), ch: 2, plugin: soundcraft_dsp::create("gain").unwrap() }],
        ))
        .unwrap();
        st.fill(&mut out, 2);
        let mut s44 = (*s).clone();
        s44.sample_rate = soundcraft_time::SampleRate::new(44_100).unwrap();
        tx.send(Cmd::Session(Arc::new(s44), Vec::new())).unwrap();
        st.fill(&mut out, 2);
        assert_eq!(grx.try_recv().unwrap().len(), 1);
    }

    #[test]
    fn plugin_states_are_read_without_the_audio_thread() {
        let mut side = Side::new(48_000.0, channel().1);
        let (a, b) = (TrackId(1), TrackId(2));
        side.states.insert((b, 0), Box::new(|| Some(b"b0".to_vec())));
        side.states.insert((a, soundcraft_mix::INSTRUMENT_SLOT), Box::new(|| Some(b"synth".to_vec())));
        side.states.insert((a, 3), Box::new(|| Some(b"a3".to_vec())));
        side.states.insert((a, 4), Box::new(|| None));
        let want = vec![(a, 3, b"a3".to_vec()), (a, soundcraft_mix::INSTRUMENT_SLOT, b"synth".to_vec()), (b, 0, b"b0".to_vec())];
        assert_eq!(side.capture_states(), want);
        side.forget((a, 3));
        assert_eq!(side.capture_states(), want[1..]);
    }

    struct Window;

    impl PluginEditor for Window {
        fn open(&mut self) -> Result<(), String> {
            Ok(())
        }
        fn close(&mut self) {}
        fn is_open(&self) -> bool {
            false
        }
        fn idle(&mut self) -> Vec<(String, f32)> {
            Vec::new()
        }
    }

    #[test]
    fn a_refused_state_is_kept_until_the_editor_opens() {
        let mut side = Side::new(48_000.0, channel().1);
        let k = (TrackId(1), 0);
        side.states.insert(k, Box::new(|| Some(b"defaults".to_vec())));
        side.editors.insert(k, Box::new(Window));
        side.refused.insert(k);
        assert!(side.capture_states().is_empty());
        side.open_editor(k).unwrap();
        assert_eq!(side.capture_states(), vec![(TrackId(1), 0, b"defaults".to_vec())]);
    }

    #[test]
    fn side_creates_each_missing_instance_once() {
        use soundcraft_model::{ChannelFormat, Insert, TrackKind};
        let mut side = Side::new(48_000.0, channel().1);
        let mut s = soundcraft_model::Session::default();
        let t = s.add_track(TrackKind::Audio, ChannelFormat::Mono, None);
        s.track_mut(t).unwrap().mixer.inserts[0] = Some(Insert::new("clap:does.not.exist"));
        s.track_mut(t).unwrap().mixer.inserts[1] = Some(Insert::new("gain"));
        assert!(side.prepare(&s).is_empty(), "the plugin is not installed; built-ins stay with the engine");
        assert_eq!(side.shadow.get(&(t, 0)), Some(&("clap:does.not.exist".to_string(), 1, false)));
        assert!(side.prepare(&s).is_empty());
        assert_eq!(side.shadow.len(), 1, "failures are remembered, not retried every update");
        s.track_mut(t).unwrap().mixer.inserts[0] = None;
        side.prepare(&s);
        assert!(side.shadow.is_empty());
    }

    #[test]
    fn replaced_sessions_are_freed_on_the_ui_thread() {
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            meters: Mutex::new(MeterSnapshot::default()),
            launch: Mutex::new(LaunchStatus::reserved()),
            played: Mutex::new(Vec::with_capacity(PLAYED_CAPACITY)),
            previewing: AtomicBool::new(false),
        });
        let (tx, rx) = channel();
        let (gtx, grx) = channel();
        let mut st = AudioState::new(rx, shared, Arc::new(soundcraft_model::Session::default()), 48_000.0, gtx);
        let mut side = Side::new(48_000.0, grx);
        side.return_old_sessions(&tx);
        let mut out = vec![0.0f32; 256];
        let mut watched = Vec::new();
        for _ in 0..3 * OLD_SESSIONS {
            let s = Arc::new(soundcraft_model::Session::default());
            watched.push(Arc::downgrade(&s));
            side.collect_garbage();
            tx.send(Cmd::Session(s, Vec::new())).unwrap();
            st.fill(&mut out, 2);
            let replaced = watched.len().checked_sub(2).and_then(|i| watched.get(i));
            assert!(replaced.is_none_or(|w| w.strong_count() == 1), "freed on the audio thread");
        }
        side.collect_garbage();
        assert_eq!(watched.iter().filter(|w| w.strong_count() > 0).count(), 1, "only the live snapshot is left");
    }

    #[test]
    fn looping_wraps() {
        let s = Arc::new(soundcraft_model::Session::default());
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            meters: Mutex::new(MeterSnapshot::default()),
            launch: Mutex::new(LaunchStatus::reserved()),
            played: Mutex::new(Vec::with_capacity(PLAYED_CAPACITY)),
            previewing: AtomicBool::new(false),
        });
        let (tx, rx) = channel();
        let mut st = AudioState::new(rx, Arc::clone(&shared), s, 48_000.0, channel().0);
        tx.send(Cmd::Play { from: 0, end: None, looped: Some(Range::new(0, 1000)) }).unwrap();
        let mut out = vec![0.0f32; 20_000];
        st.fill(&mut out, 2);
        let p = shared.position.load(Ordering::Relaxed);
        assert!(p <= 1000, "position {p} should stay inside the loop");
        assert!(shared.playing.load(Ordering::Relaxed));
    }

    #[test]
    fn locate_while_playing_drops_the_old_position_audio() {
        use soundcraft_model::{ChannelFormat, Clip, Insert, SourceAudio, SourceId, TrackKind};
        for (plugin, params) in [("compressor", &[("lookahead", 5.0), ("ratio", 1.0)][..]), ("room_reverb", &[][..])] {
            let mut s = soundcraft_model::Session::default();
            s.edit.click = false;
            let t = s.add_track(TrackKind::Audio, ChannelFormat::Mono, None);
            s.pool.insert(
                SourceId(5),
                Arc::new(SourceAudio::new(soundcraft_audio_io::AudioBuffer { sample_rate: 48_000, channels: vec![vec![0.5; 4800]] })),
            );
            let id = s.new_clip_id();
            let tr = s.track_mut(t).unwrap();
            tr.playlist_mut().unwrap().clips.push(Clip::audio(id, "x", SourceId(5), 0, 0, 4800));
            let mut ins = Insert::new(plugin);
            for (k, v) in params {
                ins.params.insert((*k).into(), *v);
            }
            tr.mixer.inserts[0] = Some(ins);
            let shared = Arc::new(Shared {
                position: AtomicI64::new(0),
                playing: AtomicBool::new(false),
                speed: AtomicU32::new(1000),
                meters: Mutex::new(MeterSnapshot::default()),
                launch: Mutex::new(LaunchStatus::reserved()),
                played: Mutex::new(Vec::with_capacity(PLAYED_CAPACITY)),
                previewing: AtomicBool::new(false),
            });
            let (tx, rx) = channel();
            let mut st = AudioState::new(rx, Arc::clone(&shared), Arc::new(s), 48_000.0, channel().0);
            tx.send(Cmd::Play { from: 0, end: None, looped: None }).unwrap();
            let mut out = vec![0.0f32; 2 * 512];
            let mut before = 0.0f32;
            for _ in 0..9 {
                st.fill(&mut out, 2);
                before = out.iter().fold(before, |m, x| m.max(x.abs()));
            }
            assert!(before > 0.1, "{plugin}: the clip plays before the locate");
            tx.send(Cmd::Locate(1_000_000)).unwrap();
            for _ in 0..8 {
                st.fill(&mut out, 2);
                assert!(out.iter().all(|x| x.abs() < 1e-6), "{plugin}: audio from before the locate");
            }
        }
    }

    #[test]
    fn armed_tracks_are_heard_and_metered_after_playback_stops() {
        use soundcraft_model::{ChannelFormat, TrackKind};
        let mut s = soundcraft_model::Session::default();
        let t = s.add_track(TrackKind::Audio, ChannelFormat::Mono, None);
        s.track_mut(t).unwrap().mixer.record_arm = true;
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            meters: Mutex::new(MeterSnapshot::default()),
            launch: Mutex::new(LaunchStatus::reserved()),
            played: Mutex::new(Vec::with_capacity(PLAYED_CAPACITY)),
            previewing: AtomicBool::new(false),
        });
        let (tx, rx) = channel();
        let mut st = AudioState::new(rx, Arc::clone(&shared), Arc::new(s), 48_000.0, channel().0);
        let ring = record::InputRing::new(1, 48_000);
        tx.send(Cmd::Input(Arc::clone(&ring))).unwrap();
        tx.send(Cmd::Play { from: 0, end: Some(1000), looped: None }).unwrap();
        let mut out = vec![0.0f32; 4096];
        st.fill(&mut out, 2);
        assert!(!shared.playing.load(Ordering::Relaxed));
        ring.push(&[0.5; 4096]);
        st.fill(&mut out, 2);
        assert!(out.iter().any(|x| x.abs() > 0.1), "the input is heard");
        assert!(shared.meters.lock().unwrap().tracks.get(&t).is_some_and(|m| m.peak[0] > 0.1), "and metered");
    }

    fn shared_for_tests() -> Arc<Shared> {
        Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            meters: Mutex::new(MeterSnapshot::default()),
            launch: Mutex::new(LaunchStatus::reserved()),
            played: Mutex::new(Vec::with_capacity(PLAYED_CAPACITY)),
            previewing: AtomicBool::new(false),
        })
    }

    #[test]
    fn live_midi_reaches_the_tracks_listening_to_its_device_and_is_kept_while_recording() {
        use soundcraft_model::{ChannelFormat, Route, TrackKind};
        let mut s = soundcraft_model::Session::default();
        let keys = s.add_track(TrackKind::Instrument, ChannelFormat::Stereo, None);
        let pad = s.add_track(TrackKind::Instrument, ChannelFormat::Stereo, None);
        let idle = s.add_track(TrackKind::Instrument, ChannelFormat::Stereo, None);
        for (t, input) in [(keys, Route::Midi("KeyStep".into())), (pad, Route::Midi("keyboard".into())), (idle, Route::None)] {
            let m = &mut s.track_mut(t).unwrap().mixer;
            m.input = input;
            m.record_arm = t != idle;
        }
        let shared = shared_for_tests();
        let (tx, rx) = channel();
        let mut st = AudioState::new(rx, Arc::clone(&shared), Arc::new(s), 48_000.0, channel().0);
        tx.send(Cmd::MidiDevices(Arc::new(vec!["Launchkey".into(), "KeyStep".into()]))).unwrap();
        tx.send(Cmd::Midi { device: 1, data: [0x90, 60, 100] }).unwrap();
        tx.send(Cmd::Midi { device: KEYBOARD, data: [0x91, 64, 90] }).unwrap();
        tx.send(Cmd::Midi { device: 0, data: [0xB0, 1, 64] }).unwrap();
        st.drain();
        let got: Vec<(TrackId, u8, u8)> = st.mix.live_notes.iter().map(|n| (n.track, n.pitch, n.velocity)).collect();
        assert_eq!(got, vec![(keys, 60, 100), (pad, 64, 90)], "by device; unarmed tracks and controllers get nothing");
        assert!(shared.played.lock().unwrap().is_empty(), "not recording: nothing kept");
        // Recording: kept with the transport position.
        st.mix.live_notes.clear();
        st.recording = true;
        st.playing = true;
        st.pos = 4_800;
        tx.send(Cmd::Midi { device: 1, data: [0x80, 60, 0] }).unwrap();
        st.drain();
        assert_eq!(*shared.played.lock().unwrap(), vec![PlayedNote { track: keys, at: 4_800, pitch: 60, velocity: 0 }]);
    }

    #[test]
    fn a_preview_plays_over_a_stopped_transport_and_ends_by_itself() {
        let shared = shared_for_tests();
        let (tx, rx) = channel();
        let mut s = soundcraft_model::Session::default();
        s.edit.click = false;
        let mut st = AudioState::new(rx, Arc::clone(&shared), Arc::new(s), 48_000.0, channel().0);
        let mut out = vec![0.0f32; 2048];
        st.fill(&mut out, 2);
        assert!(out.iter().all(|x| *x == 0.0), "stopped: silence");
        // 1500 frames at 24 kHz: 3000 at the session's 48 kHz.
        tx.send(Cmd::Preview(Some(Arc::new(PreviewSound { channels: vec![vec![0.5; 1500]], rate: 24_000 })))).unwrap();
        st.fill(&mut out, 2);
        assert!(shared.previewing.load(Ordering::Relaxed));
        assert!(out.chunks(2).take(1000).all(|f| (f[0] - 0.5).abs() < 1e-4 && (f[1] - 0.5).abs() < 1e-4), "mono on both channels");
        for _ in 0..4 {
            st.fill(&mut out, 2);
        }
        assert!(!shared.previewing.load(Ordering::Relaxed), "it ends at its last frame");
        st.fill(&mut out, 2);
        assert!(out.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn a_tap_records_a_track_before_its_fader_only_while_recording() {
        use soundcraft_model::{ChannelFormat, TrackKind};
        let mut s = soundcraft_model::Session::default();
        s.edit.click = false;
        let src = s.add_track(TrackKind::Audio, ChannelFormat::Mono, None);
        let dst = s.add_track(TrackKind::Audio, ChannelFormat::Mono, None);
        let id = soundcraft_model::SourceId(1);
        s.pool.insert(
            id,
            Arc::new(soundcraft_model::SourceAudio::new(soundcraft_audio_io::AudioBuffer { sample_rate: 48_000, channels: vec![vec![0.5; 48_000]] })),
        );
        let cid = s.new_clip_id();
        s.track_mut(src).unwrap().playlist_mut().unwrap().clips.push(soundcraft_model::Clip::audio(cid, "dc", id, 0, 0, 48_000));
        s.track_mut(src).unwrap().mixer.volume_db = -144.0;
        let shared = shared_for_tests();
        let (tx, rx) = channel();
        let mut st = AudioState::new(rx, Arc::clone(&shared), Arc::new(s), 48_000.0, channel().0);
        let dir = std::env::temp_dir().join(format!("soundcraft-tap-play-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("print.wav");
        let (feed, writer) = record::start_taps(vec![(dst, record::TapSource::Track(src), 1, path.clone())], 48_000).unwrap();
        tx.send(Cmd::Taps(Some(feed))).unwrap();
        tx.send(Cmd::Play { from: 0, end: None, looped: None }).unwrap();
        let mut out = vec![0.0f32; 2048];
        st.fill(&mut out, 2);
        tx.send(Cmd::Recording(true)).unwrap();
        st.fill(&mut out, 2);
        st.fill(&mut out, 2);
        let (done, problem) = writer.finish();
        assert_eq!((done.len(), problem), (1, None));
        let audio = soundcraft_audio_io::decode(&std::fs::read(&path).unwrap(), None).unwrap().1.channels;
        let n = audio[0].len();
        assert!(n >= 1024 && audio[0].iter().all(|x| (x - 0.5).abs() < 1e-6), "{n} frames of the source, its fader all the way down");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
