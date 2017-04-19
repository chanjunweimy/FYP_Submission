@ECHO OFF

python audioAnalysis.py  featureExtractionDir -i dcapswoz_audio_participantonly_merged/dev/ -mw 1.0 -ms 1.0 -sw 0.050 -ss 0.050
python audioAnalysis.py  featureExtractionDir -i dcapswoz_audio_participantonly_merged/train/ -mw 1.0 -ms 1.0 -sw 0.050 -ss 0.050
