# # main.py
# import logging
# from batch_scanner import BatchIngestionScanner
# from metrics_extractor import CallMetricsExtractor
# from audio_parser import StereoAudioAnalyzer

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
# )
# logger = logging.getLogger("EntropyMain")

# def run_pipeline():
#     TARGET_BATCH_DIR = "./calls"
#     scanner = BatchIngestionScanner(batch_dir=TARGET_BATCH_DIR)
    
#     logger.info(f"Starting batch pipeline on: {TARGET_BATCH_DIR}")
#     success_count = 0
    
#     for folder_path, record in scanner.scan_folders():
#         # Step 3: Core & Financial Metrics
#         extractor = CallMetricsExtractor(record)
#         operational_summary = extractor.extract_operational_summary()
#         financial_costs = extractor.calculate_costs()
        
#         # Step 4: Stereo Audio Truth & Latencies
#         wav_filename = record.artifacts.recording if (record.artifacts and record.artifacts.recording) else "recording.wav"
#         audio_analyzer = StereoAudioAnalyzer(folder_path / wav_filename)
#         audio_metrics = audio_analyzer.analyze(record)
        
#         logger.info(
#             f"Call: {record.call_id} | "
#             f"WAV Duration: {audio_metrics['wav_duration_s']}s | "
#             f"Audio Verified: {audio_metrics['audio_verified']} | "
#             f"Ghost Turns: {audio_metrics['ghost_turns_count']} | "
#             f"Acoustic P50 Latency: {audio_metrics['acoustic_latency_ms_p50']}ms"
#         )
        
#         scanner.mark_processed(record.call_id)
#         success_count += 1

#     logger.info(f"Pipeline complete. Processed {success_count} calls.")

# if __name__ == "__main__":
#     run_pipeline()

# main.py
import logging
from pathlib import Path

# Import components from each step of your pipeline
from batch_scanner import BatchIngestionScanner
from metrics_extractor import CallMetricsExtractor
from audio_parser import StereoAudioAnalyzer
from classifier import CallStoryClassifier
from sink import JSONLSink

# Configure global logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("EntropyMain")

def run_pipeline():
    """
    Orchestrates the complete ENTROPY production-grade observability pipeline:
    1. Scans and validates local call folders (with state idempotency).
    2. Extracts operational summaries and granular financial costs.
    3. Analyzes stereo audio tracks for ghost audio and true acoustic latencies.
    4. Classifies the macro narrative call story.
    5. Writes the fully enriched, standardized record to the JSONL sink.
    """
    TARGET_BATCH_DIR = "./calls"
    OUTPUT_REPORT_PATH = "outputs/batch_report.jsonl"
    
    # Initialize scanner and JSONL sink
    scanner = BatchIngestionScanner(batch_dir=TARGET_BATCH_DIR)
    sink = JSONLSink(output_path=OUTPUT_REPORT_PATH)
    
    logger.info(f"Starting ENTROPY pipeline on batch directory: {TARGET_BATCH_DIR}")
    success_count = 0
    error_count = 0
    
    # Scan folders (force_reprocess=True ensures full batch runs during testing)
    for folder_path, record in scanner.scan_folders():
        try:
            # --- Step 3: Core & Financial Metrics ---
            extractor = CallMetricsExtractor(record)
            operational_summary = extractor.extract_operational_summary()
            financial_costs = extractor.calculate_costs()
            
            # --- Step 4: Stereo Audio Truth & Latencies ---
            wav_filename = record.artifacts.recording if (record.artifacts and record.artifacts.recording) else "recording.wav"
            audio_analyzer = StereoAudioAnalyzer(folder_path / wav_filename)
            audio_metrics = audio_analyzer.analyze(record)
            
            # --- Step 5: Call Story Classification ---
            classifier = CallStoryClassifier(record, audio_metrics)
            call_story = classifier.classify()
            
            # --- Step 6: Unified Enriched Record Construction & Sink Write ---
            unified_record = {
                **operational_summary,
                "call_story": call_story,
                "financials": financial_costs,
                "audio_telemetry": audio_metrics,
                "started_at": record.started_at.isoformat() if record.started_at else None,
                "ended_at": record.ended_at.isoformat() if record.ended_at else None,
            }
            
            # Write to disk via append-safe JSONL sink
            sink.write_record(unified_record)
            
            # Mark as processed in local state tracker
            scanner.mark_processed(record.call_id)
            success_count += 1
            
            logger.info(
                f"Call ID: {record.call_id[:8]}... | "
                f"Story: {call_story:<18} | "
                f"Cost: ${financial_costs['cost_total']:<7.4f} | "
                f"Ghost Turns: {audio_metrics['ghost_turns_count']} | "
                f"Acoustic P50: {audio_metrics['acoustic_latency_ms_p50']}ms"
            )
            
        except Exception as e:
            error_count += 1
            logger.error(f"Failed processing call folder {folder_path.name}: {e}")

    logger.info(
        f"Pipeline execution finished! Successfully processed {success_count} calls "
        f"({error_count} errors). Report saved to {OUTPUT_REPORT_PATH}"
    )

if __name__ == "__main__":
    run_pipeline()