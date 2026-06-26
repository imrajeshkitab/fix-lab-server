-- ============================================================================
-- Phase 2b — Per-segment audio storage on bite_audio_triage
-- ============================================================================
-- Adds three columns to track regenerated audio segments and the final
-- stitched/uploaded audio. Designed for resumability:
--
-- segment_audio: array of generated paragraph audio (or "full") with their
--                storage URLs. If splicing fails, this preserves TTS work
--                so the next retry can skip re-paying for TTS.
--
-- final_audio_url: the final audio that landed in bites.audio[lang].url
-- final_audio_round: the round number the final audio was uploaded to
--
-- Storage convention:
--   RMS-content/bites/audio_segments_triage/{triage_id}_p{N}.mp3   (per paragraph)
--   RMS-content/bites/audio_segments_triage/{triage_id}_full.mp3   (when decision=full)
--
-- This migration is idempotent.
-- ============================================================================

ALTER TABLE bite_audio_triage
    ADD COLUMN IF NOT EXISTS segment_audio    JSONB DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS final_audio_url  TEXT,
    ADD COLUMN IF NOT EXISTS final_audio_round INT;

COMMENT ON COLUMN bite_audio_triage.segment_audio IS
    'Array of regenerated audio segments. Each entry: {paragraph_index, url, duration_sec, char_count, voice_id, generated_at}. paragraph_index=-1 indicates the full audio (used when decision=full).';

COMMENT ON COLUMN bite_audio_triage.final_audio_url IS
    'Final stitched/uploaded audio URL that was written to bites.audio[lang].url. Audit trail for "what triage produced this audio".';

COMMENT ON COLUMN bite_audio_triage.final_audio_round IS
    'Round number assigned to final_audio_url (matches bites.audio_version[lang] after the update).';
