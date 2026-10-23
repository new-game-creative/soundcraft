//! Playing the clip launcher: which slot each track is playing (transport state, set by the
//! playback host) and rendering a launched slot, which repeats every loop from its launch point.

use super::{PluginSlot, render_clip_at};
use soundcraft_model::{ClipContent, LaunchSlot, Session, TrackId};
use soundcraft_time::Samples;

/// What the launcher is playing. Owned by the playback host and handed to the mix engine.
#[derive(Debug, Clone, Default)]
pub struct LaunchPlayback {
    /// Tracks play their launched slot instead of the timeline (and are silent without one).
    pub session_mode: bool,
    /// Playing slots: (track, scene, sample the launch started at). At most one per track.
    pub playing: Vec<(TrackId, usize, Samples)>,
}

impl LaunchPlayback {
    /// The scene and start of what `track` is playing.
    pub fn playing(&self, track: TrackId) -> Option<(usize, Samples)> {
        self.playing.iter().find(|(t, _, _)| *t == track).map(|(_, scene, start)| (*scene, *start))
    }

    /// Starts `scene` on `track` at `at` (replacing what it played), or stops it with `None`.
    /// Within the capacity the host reserves, this does not allocate.
    pub fn set(&mut self, track: TrackId, scene: Option<usize>, at: Samples) {
        self.playing.retain(|(t, _, _)| *t != track);
        if let Some(scene) = scene {
            self.playing.push((track, scene, at));
        }
    }
}

/// The loop pass of a slot launched at `start` that contains sample `at`: its first sample and
/// the sample after its last. `None` before the launch, or after a one-shot slot's single pass.
pub fn iteration(s: &Session, slot: &LaunchSlot, start: Samples, at: Samples) -> Option<(Samples, Samples)> {
    if at < start || slot.loop_ticks <= 0 {
        return None;
    }
    let sr = s.sample_rate;
    let t0 = s.tempo.samples_to_ticks(start, sr);
    let edge = |k: i64| if k == 0 { start } else { s.tempo.tick_to_samples(t0.saturating_add(k.saturating_mul(slot.loop_ticks)), sr) };
    let mut k = s.tempo.samples_to_ticks(at, sr).saturating_sub(t0).div_euclid(slot.loop_ticks).max(0);
    // Sample ↔ tick rounding can put `at` a sample either side of the computed pass.
    if k > 0 && at < edge(k) {
        k -= 1;
    } else if at >= edge(k.saturating_add(1)) {
        k = k.saturating_add(1);
    }
    if k > 0 && !slot.looped {
        return None;
    }
    let (a, b) = (edge(k), edge(k.saturating_add(1)));
    (b > a).then_some((a, b))
}

/// Calls `f(pass_start, block_offset, len)` for each piece of `[pos, pos+frames)` inside a loop
/// pass of the slot (a block that crosses a loop's end is split there).
fn passes(s: &Session, slot: &LaunchSlot, start: Samples, pos: Samples, frames: usize, mut f: impl FnMut(Samples, Samples, usize, usize)) {
    let end = pos.saturating_add(frames as i64);
    let mut at = pos.max(start);
    // Bounded: a block crosses at most a few passes (loops are at least a sixteenth note).
    for _ in 0..64 {
        if at >= end {
            break;
        }
        let Some((a, b)) = iteration(s, slot, start, at) else { break };
        let seg_end = b.min(end);
        let off = usize::try_from(at - pos).unwrap_or(0);
        let len = usize::try_from(seg_end - at).unwrap_or(0);
        if len == 0 {
            break;
        }
        f(a, b, off, len);
        at = seg_end;
    }
}

/// Adds the launched slot's audio for `[pos, pos+frames)` into `buf`.
pub fn render_audio(s: &Session, track: TrackId, launch: &LaunchPlayback, pos: Samples, frames: usize, buf: &mut [Vec<f32>]) {
    let Some((scene, start)) = launch.playing(track) else { return };
    let Some(slot) = s.launcher.slot(track, scene) else { return };
    if slot.clip.muted || !slot.clip.is_audio() {
        return;
    }
    passes(s, slot, start, pos, frames, |pass, _, off, len| {
        // The clip sits at 0 in its slot: render it at the position within this pass.
        let local = pos.saturating_add(off as i64).saturating_sub(pass);
        render_clip_at(s, &slot.clip, local, len, buf, off);
    });
}

/// Sends the launched MIDI slot's note ons and offs for `[pos, pos+frames)`. `last` is what this
/// track played on the previous block: when it changes, held notes are released first.
pub fn schedule_notes(
    s: &Session,
    track: TrackId,
    launch: &LaunchPlayback,
    pos: Samples,
    frames: usize,
    inst: &mut PluginSlot,
    held: &mut Vec<u8>,
    last: &mut Option<(usize, Samples)>,
) {
    let now = launch.playing(track);
    if now != *last {
        for p in held.drain(..) {
            inst.plugin.note_off(0, p);
        }
        *last = now;
    }
    let Some((scene, start)) = now else { return };
    let Some(slot) = s.launcher.slot(track, scene) else { return };
    let ClipContent::Midi { sequence } = &slot.clip.content else { return };
    if slot.clip.muted {
        return;
    }
    let sr = s.sample_rate;
    let end = pos.saturating_add(frames as i64);
    passes(s, slot, start, pos, frames, |pass, pass_end, _, _| {
        let base = s.tempo.samples_to_ticks(pass, sr);
        // The clip is cut at the loop's end and at its own length.
        let stop = pass_end.min(pass.saturating_add(slot.clip.length));
        for n in &sequence.notes {
            let on = s.tempo.tick_to_samples(base.saturating_add(n.start), sr);
            if on >= stop {
                continue;
            }
            let off = s.tempo.tick_to_samples(base.saturating_add(n.start).saturating_add(n.length.max(1)), sr).min(stop);
            if on >= pos && on < end {
                inst.plugin.note_on(usize::try_from(on - pos).unwrap_or(0), n.pitch, n.velocity);
                if !held.contains(&n.pitch) && held.len() < 128 {
                    held.push(n.pitch);
                }
            }
            // A note running into the loop's end is released there (the frame before, so the
            // next pass can start the same pitch again).
            let off = if off == stop { off.saturating_sub(1).max(on) } else { off };
            if off >= pos && off < end {
                inst.plugin.note_off(usize::try_from(off - pos).unwrap_or(0), n.pitch);
                held.retain(|p| *p != n.pitch);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use soundcraft_model::{Clip, ClipId, SourceId};

    /// 120 BPM 4/4 at 48 kHz: a bar is 96 000 samples.
    fn session() -> Session {
        let mut s = Session::default();
        s.launcher.insert_scene(0, "A".into());
        s
    }

    fn slot(loop_bars: i64, looped: bool) -> LaunchSlot {
        let mut l = LaunchSlot::new(Clip::audio(ClipId(1), "x", SourceId(1), 0, 0, 96_000), loop_bars * 3840);
        l.looped = looped;
        l
    }

    #[test]
    fn passes_repeat_every_loop_from_the_launch() {
        let s = session();
        let sl = slot(1, true);
        assert_eq!(iteration(&s, &sl, 96_000, 50), None, "before the launch");
        assert_eq!(iteration(&s, &sl, 96_000, 96_000), Some((96_000, 192_000)));
        assert_eq!(iteration(&s, &sl, 96_000, 191_999), Some((96_000, 192_000)));
        assert_eq!(iteration(&s, &sl, 96_000, 192_000), Some((192_000, 288_000)));
        assert_eq!(iteration(&s, &slot(1, false), 96_000, 192_000), None, "a one-shot plays once");
    }

    #[test]
    fn a_block_across_the_loop_end_is_split_there() {
        let s = session();
        let sl = slot(1, true);
        let mut seen = Vec::new();
        passes(&s, &sl, 0, 95_900, 512, |pass, _, off, len| seen.push((pass, off, len)));
        assert_eq!(seen, vec![(0, 0, 100), (96_000, 100, 412)]);
    }

    /// Peak of each half bar (48 000 samples at 120 BPM) of `bars` bars rendered in 1 000-sample
    /// blocks, so blocks cross the loop ends.
    fn half_bar_peaks(s: &Session, e: &mut crate::MixEngine, bars: usize) -> Vec<f32> {
        let mut out = vec![vec![0.0f32; 1000]; 2];
        let mut peaks = vec![0.0f32; bars * 2];
        for b in 0..(bars * 96) {
            let pos = b as i64 * 1000;
            e.render(s, pos, 1000, &mut out);
            let p = out.iter().flat_map(|c| c.iter()).fold(0.0f32, |m, x| m.max(x.abs()));
            if let Some(slot) = peaks.get_mut(b / 48) {
                *slot = slot.max(p);
            }
        }
        peaks
    }

    #[test]
    fn a_launched_slot_loops_from_its_launch_and_replaces_the_timeline() {
        use soundcraft_model::{ChannelFormat, SourceAudio, TrackKind};
        let mut s = session();
        s.edit.click = false;
        let t = s.add_track(TrackKind::Audio, ChannelFormat::Mono, None);
        let dc = vec![0.5f32; 480_000];
        s.pool
            .insert(SourceId(1), std::sync::Arc::new(SourceAudio::new(soundcraft_audio_io::AudioBuffer { sample_rate: 48_000, channels: vec![dc] })));
        // On the timeline: a clip over the whole range, which session mode must not play.
        let tl = s.new_clip_id();
        s.track_mut(t).unwrap().playlist_mut().unwrap().clips.push(Clip::audio(tl, "timeline", SourceId(1), 0, 0, 480_000));
        // In the slot: half a bar of audio in a one-bar loop.
        let id = s.new_clip_id();
        s.launcher.set_slot(t, 0, Some(LaunchSlot::new(Clip::audio(id, "loop", SourceId(1), 0, 0, 48_000), 3840))).unwrap();
        let mut e = crate::MixEngine::new(48_000.0, 1024);
        e.launch.session_mode = true;
        // Launched at bar 2: silent before, then sound / silence every half bar.
        e.launch.set(t, Some(0), 96_000);
        let peaks = half_bar_peaks(&s, &mut e, 4);
        let on: Vec<bool> = peaks.iter().map(|p| *p > 0.1).collect();
        assert_eq!(on, [false, false, true, false, true, false, true, false], "{peaks:?}");
        // Stopped: silent, timeline included.
        e.launch.set(t, None, 0);
        let mut e2 = crate::MixEngine::new(48_000.0, 1024);
        e2.launch = e.launch.clone();
        assert!(half_bar_peaks(&s, &mut e2, 1).iter().all(|p| *p == 0.0));
        // Session mode off: the timeline plays again.
        let mut e3 = crate::MixEngine::new(48_000.0, 1024);
        assert!(half_bar_peaks(&s, &mut e3, 1).iter().all(|p| *p > 0.1));
    }

    #[test]
    fn launch_state_holds_one_slot_per_track() {
        let mut l = LaunchPlayback::default();
        l.set(TrackId(1), Some(0), 0);
        l.set(TrackId(1), Some(2), 96_000);
        l.set(TrackId(2), Some(1), 0);
        assert_eq!(l.playing(TrackId(1)), Some((2, 96_000)));
        l.set(TrackId(1), None, 0);
        assert_eq!(l.playing(TrackId(1)), None);
        assert_eq!(l.playing.len(), 1);
    }
}
