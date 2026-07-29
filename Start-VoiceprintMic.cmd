@echo off
setlocal

set "ORT_DYLIB_PATH=%~dp0runtime\onnxruntime.dll"
set "MELLONELLA_ECAPA_ONNX=%~dp0models\ecapa\ecapa_tdnn.onnx"
set "MELLONELLA_VAD_ONNX=%~dp0models\vad\model.onnx"
set "MELLONELLA_DFN3_ONNX=%~dp0models\dfn3\dfn3.onnx"
set "MELLONELLA_OVERLAP_SEG_ONNX=%~dp0models\overlap\model.onnx"
set "MELLONELLA_TSE_PROD_48K_ONNX=%~dp0models\tse\tse_prod_48k.onnx"

rem The legacy 8 kHz, one-second SepFormer path is intentionally disabled.
rem It split even solo speech, damaged bandwidth, and produced unstable
rem post-separation voiceprints. The 48 kHz adaptive TSE path above runs
rem only while overlapping speakers are detected.
set "MELLONELLA_SEPFORMER_ONNX="

rem Partial names are enough. The app selects the first matching device.
set "VOICEPRINTMIC_INPUT_DEVICE=HyperX QuadCast S"
set "VOICEPRINTMIC_OUTPUT_DEVICE=Realtek High Definition Audio"

start "" "%~dp0VoiceprintMic.exe"
endlocal
