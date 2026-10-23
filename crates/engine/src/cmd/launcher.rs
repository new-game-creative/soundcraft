//! The clip launcher: scenes, slots (clips that loop when launched) and launching. Editing
//! commands change the document (and undo); launching is transport, queued for the host in
//! [`Engine::launch_requests`] and carried out sample-accurately by the playback engine.

use super::*;
use crate::{LaunchRequest, TransportRequest, cmd};
use serde_json::json;
use soundcraft_model::{Clip, ClipContent, LaunchSlot, MAX_LOOP_TICKS, SourceId, TrackId};
use soundcraft_time::TICKS_PER_QUARTER;

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!("launcher.scene_add", "Add Scene", [], None, "{at?: index (end), name?}", always, scene_add),
        cmd!("launcher.scene_rename", "Rename Scene", [], None, "{scene, name}", always, scene_rename),
        cmd!("launcher.scene_remove", "Remove Scene", [], None, "{scene}", always, |e, p| {
            let scene = scene_param(e, "launcher.scene_remove", p)?;
            let removed = e.session_mut().launcher.remove_scene(scene).map(|s| s.name);
            Ok(json!({"removed": removed}))
        }),
        cmd!(
            "launcher.slot_from_clip",
            "Clip to Slot",
            [],
            None,
            "{clip: id (a timeline clip), scene, track?: (the clip's), loop_bars?: (its length in whole bars)}",
            always,
            slot_from_clip
        ),
        cmd!(
            "launcher.slot_from_source",
            "Audio to Slot",
            [],
            None,
            "{source: id | path: audio file (imported), track, scene, offset?: 0, length?: (to the end), name?, loop_bars?: (its length in whole bars)}",
            always,
            slot_from_source
        ),
        cmd!("launcher.slot_clear", "Clear Slot", [], None, "{track, scene}", always, |e, p| {
            let (t, scene) = slot_param(e, "launcher.slot_clear", p)?;
            e.session_mut().launcher.set_slot(t, scene, None);
            Ok(json!({}))
        }),
        cmd!(
            "launcher.slot_set",
            "Slot Properties",
            [],
            None,
            "{track, scene, name?, loop_bars?, looped?, gain_db?, color?: [r,g,b] | null, offset?: samples, length?: samples}",
            always,
            slot_set
        ),
        cmd!(
            "launcher.to_timeline",
            "Slot to Timeline",
            [],
            None,
            "{track, scene, start, end, launched?: (start)} — writes the slot onto its track's timeline over [start, end) as it plays when launched at `launched` (each loop pass a clip), replacing what was there",
            always,
            to_timeline
        ),
        cmd!("launcher.quantize", "Launch Quantize", [], None, "{bars: 0 (none) | 0.25 | 0.5 | 1 | 2 | 4 …}", always, |e, p| {
            let bars = p.get("bars").and_then(Value::as_f64).filter(|v| v.is_finite()).ok_or_else(|| bad("launcher.quantize", "`bars` required"))?;
            let bars = bars.clamp(0.0, 16.0);
            e.session_mut().launcher.quantize_bars = bars;
            Ok(json!({"bars": bars}))
        }),
        cmd!(noundo "launcher.launch", "Launch Slot", [], None, "{track, scene}", always, |e, p| {
            let (t, scene) = slot_param(e, "launcher.launch", p)?;
            if e.session().launcher.slot(t, scene).is_none() {
                return Err(bad("launcher.launch", "that slot is empty"));
            }
            launch(e, &[(t, Some(scene))]);
            Ok(json!({}))
        }),
        cmd!(noundo "launcher.resume", "Resume Slot", [], None, "{track, scene, start: samples} — plays now, as if launched at `start`", always, |e, p| {
            let (t, scene) = slot_param(e, "launcher.resume", p)?;
            if e.session().launcher.slot(t, scene).is_none() {
                return Err(bad("launcher.resume", "that slot is empty"));
            }
            let start = position_param(e, "launcher.resume", p, "start")?.ok_or_else(|| bad("launcher.resume", "`start` required"))?;
            e.launch_requests.push(LaunchRequest::Resume { track: t, scene, start });
            Ok(json!({}))
        }),
        cmd!(noundo "launcher.stop", "Stop Track Slot", [], None, "{track | tracks}", always, |e, p| {
            let tracks = tracks_required(e, "launcher.stop", p)?;
            let stops: Vec<(TrackId, Option<usize>)> = tracks.into_iter().map(|t| (t, None)).collect();
            launch(e, &stops);
            Ok(json!({}))
        }),
        cmd!(noundo "launcher.launch_scene", "Launch Scene", [], None, "{scene} — every track with a slot in it launches it; the others stop", always, |e, p| {
            let scene = scene_param(e, "launcher.launch_scene", p)?;
            let s = e.session();
            let changes: Vec<(TrackId, Option<usize>)> =
                s.launcher.rows.iter().map(|r| (r.track, r.slots.get(scene).and_then(Option::as_ref).map(|_| scene))).collect();
            launch(e, &changes);
            Ok(json!({"tracks": changes.len()}))
        }),
        cmd!(noundo "launcher.stop_all", "Stop All Slots", [], None, "{}", always, |e, _| {
            let stops: Vec<(TrackId, Option<usize>)> = e.session().launcher.rows.iter().map(|r| (r.track, None)).collect();
            launch(e, &stops);
            Ok(json!({}))
        }),
        cmd!(noundo "launcher.session_mode", "Session Mode", [], None, "{value: bool} — tracks play their launched slots (true) or the timeline (false)", always, |e, p| {
            let on = p.get("value").and_then(Value::as_bool).ok_or_else(|| bad("launcher.session_mode", "`value` (bool) required"))?;
            e.launch_requests.push(LaunchRequest::SessionMode(on));
            Ok(json!({"value": on}))
        }),
    ]
}

/// Queues launches (scene) and stops (`None`) at the current quantize; starts the transport when
/// it is stopped, so they begin at once.
fn launch(e: &mut Engine, changes: &[(TrackId, Option<usize>)]) {
    let quantize_bars = e.session().launcher.quantize_bars;
    if changes.iter().any(|(_, s)| s.is_some()) && !e.transport.playing && !e.transport_requests.contains(&TransportRequest::Play) {
        e.transport_requests.push(TransportRequest::Play);
    }
    for (track, scene) in changes {
        e.launch_requests.push(LaunchRequest::Launch { track: *track, scene: *scene, quantize_bars });
    }
}

fn scene_param(e: &Engine, cmd: &str, p: &Value) -> Result<usize> {
    let n = p.get("scene").and_then(Value::as_u64).ok_or_else(|| bad(cmd, "`scene` (index) required"))?;
    let n = usize::try_from(n).map_err(|_| bad(cmd, "no such scene"))?;
    if n < e.session().launcher.scenes.len() { Ok(n) } else { Err(bad(cmd, format!("no scene {n}"))) }
}

fn slot_param(e: &Engine, cmd: &str, p: &Value) -> Result<(TrackId, usize)> {
    let t = track_param(e, cmd, p, "track")?.ok_or_else(|| bad(cmd, "`track` required"))?;
    Ok((t, scene_param(e, cmd, p)?))
}

/// A loop length from `loop_bars` (whole or fractional bars, in the meter at the start), or the
/// clip's own length rounded up to whole bars.
fn loop_ticks(e: &Engine, p: &Value, clip_len: Samples) -> i64 {
    let s = e.session();
    let bar = s.tempo.meter_at_tick(0).ticks_per_bar().max(1);
    let ticks = match p.get("loop_bars").and_then(Value::as_f64).filter(|v| v.is_finite() && *v > 0.0) {
        Some(bars) => (bars * bar as f64).round() as i64,
        None => {
            let t = s.tempo.samples_to_ticks(clip_len, s.sample_rate).max(1);
            t.saturating_add(bar - 1) / bar * bar
        }
    };
    ticks.clamp(TICKS_PER_QUARTER / 4, MAX_LOOP_TICKS)
}

fn scene_add(e: &mut Engine, p: &Value) -> Result<Value> {
    let n = e.session().launcher.scenes.len();
    let at = p.get("at").and_then(Value::as_u64).and_then(|v| usize::try_from(v).ok()).unwrap_or(n);
    let name = str_param(p, "name").map(str::to_string).unwrap_or_else(|| format!("Scene {}", n + 1));
    let scene = e.session_mut().launcher.insert_scene(at, name).ok_or_else(|| bad("launcher.scene_add", "too many scenes"))?;
    Ok(json!({"scene": scene}))
}

fn scene_rename(e: &mut Engine, p: &Value) -> Result<Value> {
    let scene = scene_param(e, "launcher.scene_rename", p)?;
    let name = str_param(p, "name").filter(|n| !n.trim().is_empty()).ok_or_else(|| bad("launcher.scene_rename", "`name` required"))?.to_string();
    if let Some(sc) = e.session_mut().launcher.scenes.get_mut(scene) {
        sc.name = name;
    }
    Ok(json!({}))
}

fn put(e: &mut Engine, cmd: &str, t: TrackId, scene: usize, mut clip: Clip, ticks: i64) -> Result<Value> {
    let s = e.session_mut();
    clip.id = s.new_clip_id();
    let id = clip.id;
    s.launcher.set_slot(t, scene, Some(LaunchSlot::new(clip, ticks))).ok_or_else(|| bad(cmd, "no such scene"))?;
    Ok(json!({"track": t, "scene": scene, "clip": id}))
}

fn slot_from_clip(e: &mut Engine, p: &Value) -> Result<Value> {
    const CMD: &str = "launcher.slot_from_clip";
    let id = clip_id(p, CMD)?;
    let (owner, clip) = e.session().find_clip(id).map(|(t, c)| (t, c.clone())).ok_or_else(|| bad(CMD, format!("no clip {id}")))?;
    let t = track_param(e, CMD, p, "track")?.unwrap_or(owner);
    let scene = scene_param(e, CMD, p)?;
    let kind_ok = match (&clip.content, e.session().track(t).map(|x| x.kind)) {
        (ClipContent::Audio { .. }, Some(k)) => k == soundcraft_model::TrackKind::Audio,
        (ClipContent::Midi { .. }, Some(k)) => matches!(k, soundcraft_model::TrackKind::Midi | soundcraft_model::TrackKind::Instrument),
        _ => false,
    };
    if !kind_ok {
        return Err(bad(CMD, "audio clips go on audio tracks, MIDI clips on MIDI or instrument tracks"));
    }
    let ticks = loop_ticks(e, p, clip.length);
    put(e, CMD, t, scene, clip, ticks)
}

fn slot_from_source(e: &mut Engine, p: &Value) -> Result<Value> {
    const CMD: &str = "launcher.slot_from_source";
    let (t, scene) = slot_param(e, CMD, p)?;
    if e.session().track(t).map(|x| x.kind) != Some(soundcraft_model::TrackKind::Audio) {
        return Err(bad(CMD, "audio goes on an audio track"));
    }
    // A file is imported first (one step with the slot).
    let source = match (p.get("source").and_then(Value::as_u64), str_param(p, "path")) {
        (Some(id), _) => SourceId(id),
        (None, Some(path)) => crate::io::import_source(e, path)?,
        (None, None) => return Err(bad(CMD, "`source` id or `path` required")),
    };
    let s = e.session();
    let src = s.sources.iter().find(|x| x.id == source).ok_or_else(|| bad(CMD, format!("no source {}", source.0)))?;
    if s.track(t).map(|x| x.kind) != Some(soundcraft_model::TrackKind::Audio) {
        return Err(bad(CMD, "audio goes on an audio track"));
    }
    let total = i64::try_from(src.frames).unwrap_or(i64::MAX);
    let offset = i64_or(p, "offset", 0).clamp(0, total.max(0));
    let length = i64_or(p, "length", total - offset).clamp(1, (total - offset).max(1));
    let name = str_param(p, "name").map_or_else(|| src.name.clone(), str::to_string);
    let clip = Clip::audio(soundcraft_model::ClipId(0), name, source, offset, 0, length);
    let ticks = loop_ticks(e, p, length);
    put(e, CMD, t, scene, clip, ticks)
}

/// Captures a launched slot into the arrangement: the timeline then plays what the launcher did.
fn to_timeline(e: &mut Engine, p: &Value) -> Result<Value> {
    const CMD: &str = "launcher.to_timeline";
    let (t, scene) = slot_param(e, CMD, p)?;
    let start = position_param(e, CMD, p, "start")?.ok_or_else(|| bad(CMD, "`start` required"))?.max(0);
    let end = position_param(e, CMD, p, "end")?.ok_or_else(|| bad(CMD, "`end` required"))?;
    if end <= start {
        return Err(bad(CMD, "`end` must be after `start`"));
    }
    let launched = position_param(e, CMD, p, "launched")?.unwrap_or(start);
    let s = e.session();
    let slot = s.launcher.slot(t, scene).ok_or_else(|| bad(CMD, "that slot is empty"))?.clone();
    if s.track(t).and_then(|x| x.playlist()).is_none() {
        return Err(bad(CMD, "the track has no playlist"));
    }
    // Each loop pass is the slot's clip placed at the pass start, cut at the pass end and to the range.
    let mut pieces = Vec::new();
    let mut at = start.max(launched);
    while at < end && pieces.len() < 4096 {
        let Some((a, b)) = soundcraft_mix::launch_iteration(s, &slot, launched, at) else { break };
        let mut clip = slot.clip.clone();
        clip.start = a;
        clip.length = clip.length.min(b - a);
        let (lo, hi) = (at, end.min(clip.end()));
        if hi > lo {
            pieces.push((clip, lo, hi));
        }
        at = b;
    }
    let s = e.session_mut();
    crate::edit::clear_range(s, t, soundcraft_time::Range::new(start, end), false);
    let mut ids = Vec::new();
    for (clip, lo, hi) in pieces {
        let tc = crate::edit::TimeCtx::of(s);
        let mut c = clip;
        if lo > c.start
            && let Some((_, right)) = crate::edit::split_clip(&c, lo, c.id, tc)
        {
            c = right;
        }
        if hi < c.end()
            && let Some((left, _)) = crate::edit::split_clip(&c, hi, c.id, tc)
        {
            c = left;
        }
        c.id = s.new_clip_id();
        ids.push(c.id);
        crate::edit::place_clip(s, t, c);
    }
    Ok(json!({"clips": ids}))
}

fn slot_set(e: &mut Engine, p: &Value) -> Result<Value> {
    const CMD: &str = "launcher.slot_set";
    let (t, scene) = slot_param(e, CMD, p)?;
    let ticks = p.get("loop_bars").is_some().then(|| loop_ticks(e, p, 0));
    let s = e.session_mut();
    let slot = s.launcher.slot_mut(t, scene).ok_or_else(|| bad(CMD, "that slot is empty"))?;
    if let Some(name) = str_param(p, "name").filter(|n| !n.trim().is_empty()) {
        slot.clip.name = name.to_string();
    }
    if let Some(ticks) = ticks {
        slot.loop_ticks = ticks;
    }
    if let Some(v) = p.get("looped").and_then(Value::as_bool) {
        slot.looped = v;
    }
    if let Some(db) = p.get("gain_db").and_then(Value::as_f64).filter(|v| v.is_finite()) {
        slot.clip.gain_db = (db as f32).clamp(-144.0, 24.0);
    }
    match p.get("color") {
        Some(Value::Null) => slot.clip.color = None,
        Some(Value::Array(a)) if a.len() == 3 => {
            let c: Vec<u8> = a.iter().filter_map(Value::as_u64).map(|v| v.min(255) as u8).collect();
            if let [r, g, b] = c[..] {
                slot.clip.color = Some([r, g, b]);
            }
        }
        _ => {}
    }
    // Trim (audio): a new start in the source and length; bounded by the source.
    if let ClipContent::Audio { source, offset } = &mut slot.clip.content {
        let total = s.sources.iter().find(|x| x.id == *source).map_or(i64::MAX, |x| i64::try_from(x.frames).unwrap_or(i64::MAX));
        if let Some(o) = p.get("offset").and_then(Value::as_i64) {
            *offset = o.clamp(0, total.saturating_sub(1).max(0));
        }
        if let Some(l) = p.get("length").and_then(Value::as_i64) {
            slot.clip.length = l.clamp(1, total.saturating_sub(*offset).max(1));
        }
    }
    let slot = slot.clone();
    Ok(json!({"name": slot.clip.name, "loop_ticks": slot.loop_ticks, "looped": slot.looped}))
}

#[cfg(test)]
mod tests {
    use crate::{Engine, LaunchRequest};
    use serde_json::json;

    fn engine() -> (Engine, u64) {
        let e = crate::demo::demo_engine();
        let kick = e.session().tracks.iter().find(|t| t.name == "Kick").unwrap().id.0;
        (e, kick)
    }

    #[test]
    fn scenes_and_slots_edit_and_undo() {
        let (mut e, kick) = engine();
        e.execute("launcher.scene_add", &json!({"name": "Stems"})).unwrap();
        e.execute("launcher.scene_add", &json!({})).unwrap();
        assert_eq!(e.session().launcher.scenes[1].name, "Scene 2");
        let clip = e.session().track(soundcraft_model::TrackId(kick)).unwrap().clips()[0].id.0;
        let v = e.execute("launcher.slot_from_clip", &json!({"clip": clip, "scene": 1})).unwrap();
        let slot = e.session().launcher.slot(soundcraft_model::TrackId(kick), 1).unwrap();
        assert_eq!(slot.clip.start, 0);
        assert_eq!(slot.loop_ticks % 3840, 0, "whole bars by default");
        assert_ne!(v["clip"].as_u64(), Some(clip), "the slot holds a copy with its own id");
        e.execute("launcher.slot_set", &json!({"track": kick, "scene": 1, "name": "Beat A", "loop_bars": 2})).unwrap();
        let slot = e.session().launcher.slot(soundcraft_model::TrackId(kick), 1).unwrap();
        assert_eq!((slot.clip.name.as_str(), slot.loop_ticks), ("Beat A", 7680));
        e.execute("edit.undo", &json!({})).unwrap();
        assert_eq!(e.session().launcher.slot(soundcraft_model::TrackId(kick), 1).unwrap().loop_ticks % 3840, 0);
        // MIDI clips do not go on audio tracks.
        let keys = e.session().tracks.iter().find(|t| t.name == "Keys").unwrap().clips()[0].id.0;
        assert!(e.execute("launcher.slot_from_clip", &json!({"clip": keys, "scene": 0, "track": kick})).is_err());
    }

    #[test]
    fn launching_queues_requests_and_starts_the_transport() {
        let (mut e, kick) = engine();
        e.execute("launcher.scene_add", &json!({})).unwrap();
        assert!(e.execute("launcher.launch", &json!({"track": kick, "scene": 0})).is_err(), "empty slot");
        let clip = e.session().track(soundcraft_model::TrackId(kick)).unwrap().clips()[0].id.0;
        e.execute("launcher.slot_from_clip", &json!({"clip": clip, "scene": 0})).unwrap();
        e.execute("launcher.quantize", &json!({"bars": 0.5})).unwrap();
        e.transport_requests.clear();
        e.execute("launcher.launch_scene", &json!({"scene": 0})).unwrap();
        assert_eq!(e.transport_requests, vec![crate::TransportRequest::Play]);
        assert_eq!(e.launch_requests, vec![LaunchRequest::Launch { track: soundcraft_model::TrackId(kick), scene: Some(0), quantize_bars: 0.5 }]);
        // Launching is transport, not an edit: undo takes back the quantize change, not the launch.
        let before = e.session().launcher.clone();
        e.execute("edit.undo", &json!({})).unwrap();
        assert_ne!(e.session().launcher.quantize_bars, before.quantize_bars);
    }

    /// The launched slot and the clips captured from it render the same samples.
    #[test]
    fn a_slot_written_to_the_timeline_plays_as_launched() {
        let (mut e, kick) = engine();
        let kick_id = soundcraft_model::TrackId(kick);
        let clip = e.session().track(kick_id).unwrap().clips()[0].id.0;
        e.execute("launcher.scene_add", &json!({})).unwrap();
        e.execute("launcher.slot_from_clip", &json!({"clip": clip, "scene": 0, "loop_bars": 1.5})).unwrap();
        let bar = e.session().tempo.tick_to_samples(3840, e.session().sample_rate);
        // Launched mid-bar, captured from a later point: the passes keep the launch's phase.
        let (launched, start) = (bar / 3, bar * 2 + 777);
        let end = start + bar * 4;
        let solo_kick = |s: &mut soundcraft_model::Session| {
            s.edit.click = false;
            for t in &mut s.tracks {
                t.mixer.mute = t.id != kick_id;
            }
        };
        let render = |s: &soundcraft_model::Session, session_mode: bool| {
            let mut m = soundcraft_mix::MixEngine::new(s.sample_rate.as_f64() as f32, 1024);
            m.launch.session_mode = session_mode;
            m.launch.set(kick_id, session_mode.then_some(0), launched);
            let mut out = vec![vec![0.0f32; 1000]; 2];
            let mut all = Vec::new();
            let mut pos = start;
            while pos < end {
                m.render(s, pos, 1000, &mut out);
                all.extend_from_slice(&out[0]);
                pos += 1000;
            }
            all.truncate(usize::try_from(end - start).unwrap());
            all
        };
        let mut s = (*e.session_arc()).clone();
        solo_kick(&mut s);
        let live = render(&s, true);
        assert!(live.iter().any(|x| x.abs() > 0.01), "the slot is audible");

        e.execute("launcher.to_timeline", &json!({"track": kick, "scene": 0, "start": start, "end": end, "launched": launched})).unwrap();
        let mut s2 = (*e.session_arc()).clone();
        solo_kick(&mut s2);
        let captured = render(&s2, false);
        let worst = live.iter().zip(&captured).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(worst < 1e-5, "differs by {worst}");

        e.execute("edit.undo", &json!({})).unwrap();
        assert_eq!(e.session().track(kick_id).unwrap().clips(), s.track(kick_id).unwrap().clips(), "one undo step");
    }

    /// A WAV dropped on an empty slot: imported, looping, its loop whole bars at the session tempo.
    #[test]
    fn an_audio_file_becomes_a_looping_slot_in_one_step() {
        let (mut e, kick) = engine();
        e.execute("launcher.scene_add", &json!({})).unwrap();
        let dir = std::env::temp_dir().join(format!("soundcraft-slot-file-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("Loop 120.wav");
        // 1.5 s of 44.1 kHz audio: resampled to the session rate, two bars long at 104 BPM.
        let buf = soundcraft_audio_io::AudioBuffer { sample_rate: 44_100, channels: vec![vec![0.25; 66_150]] };
        std::fs::write(&path, soundcraft_audio_io::encode(&buf, &soundcraft_audio_io::EncodeOptions::default()).unwrap()).unwrap();
        let sources = e.session().sources.len();
        let r = e.execute("launcher.slot_from_source", &json!({"track": kick, "scene": 0, "path": path.to_string_lossy()})).unwrap();
        let slot = e.session().launcher.slot(soundcraft_model::TrackId(kick), 0).unwrap().clone();
        assert!(slot.looped && slot.loop_ticks % 3840 == 0 && slot.loop_ticks > 0, "{slot:?}");
        assert_eq!(slot.clip.name, "Loop 120");
        assert_eq!(e.session().sources.len(), sources + 1);
        assert_eq!(slot.clip.length, e.session().sources.last().unwrap().frames as i64, "the whole file, at the session rate");
        assert!(r["clip"].is_u64());
        e.execute("edit.undo", &json!({})).unwrap();
        assert!(e.session().launcher.slot(soundcraft_model::TrackId(kick), 0).is_none());
        assert_eq!(e.session().sources.len(), sources, "the import goes with the same undo");
        assert!(
            e.execute("launcher.slot_from_source", &json!({"track": kick, "scene": 0, "path": dir.join("missing.wav").to_string_lossy()})).is_err()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
