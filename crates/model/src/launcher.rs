//! The clip launcher: a grid of scenes × tracks of slots, each holding a clip that loops when
//! launched (a session/clip-launch view). Pure data: which slot is playing is transport state,
//! kept by the playback engine.

use crate::{Clip, TrackId};

/// Most scenes a launcher holds.
pub const MAX_SCENES: usize = 256;
/// Ticks per quarter note (as in the tempo map).
const TPQ: i64 = soundcraft_time::TICKS_PER_QUARTER;
/// Longest loop: 1024 bars of 4/4.
pub const MAX_LOOP_TICKS: i64 = TPQ * 4 * 1024;

/// The launcher. Empty (the default) in sessions that never used it, and then not saved.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Launcher {
    #[serde(default)]
    pub scenes: Vec<Scene>,
    /// Slots by track; `rows[i].slots[j]` belongs to `scenes[j]`.
    #[serde(default)]
    pub rows: Vec<LaunchRow>,
    /// Launches and stops wait for the next multiple of this many bars (counted from bar 1);
    /// 0 starts them at once.
    #[serde(default = "one")]
    pub quantize_bars: f64,
}

impl Default for Launcher {
    fn default() -> Self {
        Launcher { scenes: Vec::new(), rows: Vec::new(), quantize_bars: 1.0 }
    }
}

fn one() -> f64 {
    1.0
}

fn yes() -> bool {
    true
}

/// A row of slots across the tracks, launched together.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Scene {
    pub name: String,
}

/// One track's slots.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LaunchRow {
    pub track: TrackId,
    #[serde(default)]
    pub slots: Vec<Option<LaunchSlot>>,
    /// How a slot launched while another plays on this track starts.
    #[serde(default, skip_serializing_if = "is_default")]
    pub launch_mode: LaunchMode,
    /// This track's launch quantize in bars (0 = at once); `None` follows
    /// [`Launcher::quantize_bars`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub launch_quantize: Option<f64>,
}

impl LaunchRow {
    pub fn new(track: TrackId) -> Self {
        LaunchRow { track, slots: Vec::new(), launch_mode: LaunchMode::default(), launch_quantize: None }
    }

    /// True when the row holds no slot and keeps the default launch settings.
    pub fn is_empty(&self) -> bool {
        self.slots.iter().all(Option::is_none) && self.launch_mode == LaunchMode::default() && self.launch_quantize.is_none()
    }
}

fn is_default<T: Default + PartialEq>(v: &T) -> bool {
    *v == T::default()
}

/// How a slot starts when it replaces one already playing on its track.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LaunchMode {
    /// From its beginning.
    #[default]
    Restart,
    /// From where the playing slot is in its loop: the new slot takes over the old one's start
    /// reference. Scene launches and launches on a stopped track still start from the top.
    Legato,
}

impl LaunchMode {
    pub fn id(self) -> &'static str {
        match self {
            LaunchMode::Restart => "restart",
            LaunchMode::Legato => "legato",
        }
    }

    pub fn from_id(s: &str) -> Option<LaunchMode> {
        [LaunchMode::Restart, LaunchMode::Legato].into_iter().find(|m| m.id().eq_ignore_ascii_case(s))
    }
}

/// A clip in a slot.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LaunchSlot {
    /// The clip (audio region or MIDI sequence, gain, fades), positioned at 0.
    pub clip: Clip,
    /// The loop, in ticks: the clip repeats every `loop_ticks`, cut at the loop's end (a shorter
    /// clip is followed by silence).
    pub loop_ticks: i64,
    /// Loops until stopped; otherwise plays once.
    #[serde(default = "yes")]
    pub looped: bool,
}

impl LaunchSlot {
    pub fn new(clip: Clip, loop_ticks: i64) -> Self {
        let mut s = LaunchSlot { clip, loop_ticks, looped: true };
        s.sanitize();
        s
    }

    fn sanitize(&mut self) {
        self.clip.start = 0;
        self.clip.sanitize();
        self.loop_ticks = self.loop_ticks.clamp(TPQ / 4, MAX_LOOP_TICKS);
    }
}

impl Launcher {
    /// True when there is nothing to save.
    pub fn is_empty(&self) -> bool {
        self.scenes.is_empty() && self.rows.iter().all(LaunchRow::is_empty) && self.quantize_bars == 1.0
    }

    /// `track`'s row, added when missing.
    pub fn row_mut(&mut self, track: TrackId) -> Option<&mut LaunchRow> {
        match self.rows.iter().position(|r| r.track == track) {
            Some(i) => self.rows.get_mut(i),
            None => {
                self.rows.push(LaunchRow::new(track));
                self.rows.last_mut()
            }
        }
    }

    /// The quantize (bars, 0 = at once) a launch on `track` waits for: the track's own, else the
    /// launcher's.
    pub fn quantize_for(&self, track: TrackId) -> f64 {
        self.row(track).and_then(|r| r.launch_quantize).unwrap_or(self.quantize_bars)
    }

    /// How a slot launched on `track` starts.
    pub fn mode_for(&self, track: TrackId) -> LaunchMode {
        self.row(track).map(|r| r.launch_mode).unwrap_or_default()
    }

    pub fn row(&self, track: TrackId) -> Option<&LaunchRow> {
        self.rows.iter().find(|r| r.track == track)
    }

    pub fn slot(&self, track: TrackId, scene: usize) -> Option<&LaunchSlot> {
        self.row(track)?.slots.get(scene)?.as_ref()
    }

    pub fn slot_mut(&mut self, track: TrackId, scene: usize) -> Option<&mut LaunchSlot> {
        self.rows.iter_mut().find(|r| r.track == track)?.slots.get_mut(scene)?.as_mut()
    }

    /// Puts `slot` (or nothing) at `track` × `scene`, adding the row and growing it as needed.
    /// `None` when `scene` is not a scene.
    pub fn set_slot(&mut self, track: TrackId, scene: usize, slot: Option<LaunchSlot>) -> Option<()> {
        if scene >= self.scenes.len() {
            return None;
        }
        let row = self.row_mut(track)?;
        if row.slots.len() <= scene {
            row.slots.resize(scene + 1, None);
        }
        *row.slots.get_mut(scene)? = slot;
        Some(())
    }

    /// Adds a scene at `at` (the end when past it), moving every row's slots with it.
    pub fn insert_scene(&mut self, at: usize, name: String) -> Option<usize> {
        if self.scenes.len() >= MAX_SCENES {
            return None;
        }
        let at = at.min(self.scenes.len());
        self.scenes.insert(at, Scene { name });
        for r in &mut self.rows {
            if r.slots.len() > at {
                r.slots.insert(at, None);
            }
        }
        Some(at)
    }

    /// Removes scene `at` and its slots.
    pub fn remove_scene(&mut self, at: usize) -> Option<Scene> {
        if at >= self.scenes.len() {
            return None;
        }
        for r in &mut self.rows {
            if r.slots.len() > at {
                r.slots.remove(at);
            }
        }
        Some(self.scenes.remove(at))
    }

    /// Every slot clip, for id bookkeeping.
    pub fn clips(&self) -> impl Iterator<Item = &Clip> {
        self.rows.iter().flat_map(|r| r.slots.iter().flatten().map(|s| &s.clip))
    }

    /// Repairs invariants after loading untrusted data: rows only for `tracks` (one each), slots
    /// no longer than the scene list, clips at 0, loop lengths and quantize in range.
    pub fn sanitize(&mut self, tracks: &[TrackId]) {
        self.scenes.truncate(MAX_SCENES);
        for s in &mut self.scenes {
            crate::truncate_chars(&mut s.name, crate::MAX_NAME_CHARS);
        }
        let mut seen = Vec::new();
        self.rows.retain(|r| {
            let keep = tracks.contains(&r.track) && !seen.contains(&r.track);
            seen.push(r.track);
            keep
        });
        let n = self.scenes.len();
        for r in &mut self.rows {
            r.slots.truncate(n);
            for s in r.slots.iter_mut().flatten() {
                s.sanitize();
            }
            for s in &mut r.slots {
                if s.as_ref().is_some_and(|x| x.clip.length <= 0) {
                    *s = None;
                }
            }
            r.launch_quantize = r.launch_quantize.filter(|q| q.is_finite()).map(|q| q.clamp(0.0, 16.0));
        }
        self.quantize_bars = if self.quantize_bars.is_finite() { self.quantize_bars.clamp(0.0, 16.0) } else { 1.0 };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ClipId, SourceId};

    fn clip(id: u64) -> Clip {
        Clip::audio(ClipId(id), "loop", SourceId(1), 0, 500, 48_000)
    }

    #[test]
    fn slots_follow_their_scenes() {
        let mut l = Launcher::default();
        assert!(l.is_empty());
        assert_eq!(l.set_slot(TrackId(1), 0, None), None, "no scene 0 yet");
        l.insert_scene(0, "Intro".into());
        l.insert_scene(1, "Verse".into());
        l.set_slot(TrackId(1), 1, Some(LaunchSlot::new(clip(9), 3840))).unwrap();
        assert!(!l.is_empty());
        assert_eq!(l.slot(TrackId(1), 1).map(|s| s.clip.start), Some(0), "slot clips sit at 0");
        // A scene inserted above moves the slot down with its scene.
        l.insert_scene(0, "Stems".into());
        assert!(l.slot(TrackId(1), 1).is_none());
        assert!(l.slot(TrackId(1), 2).is_some());
        assert_eq!(l.remove_scene(2).map(|s| s.name), Some("Verse".into()));
        assert!(l.slot(TrackId(1), 2).is_none());
    }

    #[test]
    fn sanitize_drops_rows_for_missing_tracks_and_bounds_values() {
        let mut l = Launcher::default();
        l.insert_scene(0, "A".into());
        l.set_slot(TrackId(1), 0, Some(LaunchSlot::new(clip(9), 3840))).unwrap();
        l.set_slot(TrackId(2), 0, Some(LaunchSlot::new(clip(10), 3840))).unwrap();
        l.rows.push(LaunchRow { slots: vec![None, None, None], ..LaunchRow::new(TrackId(1)) });
        if let Some(r) = l.rows.first_mut() {
            r.launch_quantize = Some(f64::INFINITY);
        }
        if let Some(s) = l.slot_mut(TrackId(1), 0) {
            s.loop_ticks = -5;
            s.clip.start = 777;
        }
        l.quantize_bars = f64::NAN;
        l.sanitize(&[TrackId(1)]);
        assert_eq!(l.rows.len(), 1);
        let s = l.slot(TrackId(1), 0).unwrap();
        assert_eq!((s.clip.start, s.loop_ticks), (0, TPQ / 4));
        assert_eq!(l.quantize_bars, 1.0);
        assert_eq!(l.rows[0].launch_quantize, None, "a bad track quantize follows the launcher's");
    }

    #[test]
    fn tracks_keep_their_own_launch_settings() {
        let mut l = Launcher::default();
        assert_eq!((l.quantize_for(TrackId(1)), l.mode_for(TrackId(1))), (1.0, LaunchMode::Restart));
        if let Some(r) = l.row_mut(TrackId(1)) {
            r.launch_mode = LaunchMode::Legato;
            r.launch_quantize = Some(0.25);
        }
        assert!(!l.is_empty(), "a track's launch settings are saved");
        assert_eq!((l.quantize_for(TrackId(1)), l.mode_for(TrackId(1))), (0.25, LaunchMode::Legato));
        assert_eq!(l.quantize_for(TrackId(2)), 1.0);
        let mut s = crate::Session::default();
        let t = s.add_track(crate::TrackKind::Audio, crate::ChannelFormat::Mono, None);
        s.launcher = l.clone();
        s.launcher.rows[0].track = t;
        let back = crate::Session::from_json(&s.to_json().unwrap()).unwrap();
        assert_eq!(back.launcher, s.launcher);
        assert_eq!(LaunchMode::from_id("LEGATO"), Some(LaunchMode::Legato));
    }

    #[test]
    fn an_empty_launcher_is_not_saved_and_old_sessions_load() {
        let s = crate::Session::default();
        let json = s.to_json().unwrap();
        assert!(!json.contains("launcher"), "unused launcher left out");
        let back = crate::Session::from_json(&json).unwrap();
        assert!(back.launcher.is_empty());
    }

    #[test]
    fn a_used_launcher_round_trips() {
        let mut s = crate::Session::default();
        let t = s.add_track(crate::TrackKind::Audio, crate::ChannelFormat::Mono, None);
        s.launcher.insert_scene(0, "Stems".into());
        let id = s.new_clip_id();
        s.launcher.set_slot(t, 0, Some(LaunchSlot::new(Clip::audio(id, "drums", SourceId(1), 0, 0, 96_000), 7680))).unwrap();
        s.launcher.quantize_bars = 0.5;
        let back = crate::Session::from_json(&s.to_json().unwrap()).unwrap();
        assert_eq!(back.launcher, s.launcher);
        // Ids stay unique after loading: the next id is past the slot clip's.
        let mut back = back;
        assert!(back.alloc() > id.0);
    }
}
