from mido import Message, MidiFile, MidiTrack

from hybrid_music_engine.transcription.midi_cleanup import cleanup_midi_layer


def test_cleanup_midi_layer_filters_range_and_quantizes(tmp_path):
    midi_path = tmp_path / "bass.mid"
    midi = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    midi.tracks.append(track)
    track.append(Message("program_change", channel=0, program=0, time=0))
    track.append(Message("note_on", channel=0, note=20, velocity=90, time=7))
    track.append(Message("note_off", channel=0, note=20, velocity=0, time=20))
    track.append(Message("note_on", channel=0, note=44, velocity=90, time=11))
    track.append(Message("note_off", channel=0, note=44, velocity=0, time=150))
    midi.save(midi_path)

    report = cleanup_midi_layer(midi_path, layer="bass", quantize_grid="1/16", min_note_ticks=60)

    assert report["valid"] is True
    cleaned = MidiFile(midi_path)
    notes = [message.note for track in cleaned.tracks for message in track if message.type == "note_on" and message.velocity > 0]
    assert notes == [44]
    absolute_tick = 0
    first_tick = None
    for message in cleaned.tracks[0]:
        absolute_tick += message.time
        if message.type == "note_on" and message.velocity > 0:
            first_tick = absolute_tick
            break
    assert first_tick is not None
    assert first_tick % 120 == 0
