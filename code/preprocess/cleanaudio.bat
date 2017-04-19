@ECHO OFF

SETLOCAL enabledelayedexpansion
for /r %%i in (*) do (    
    SET line=%%i

    IF NOT "!line:.wav=!" == "%%i" (
        echo 'getting noise sample.'

        ffmpeg -i %%i -vn -ss 00:00:00 -t 00:00:01 !line:.wav=_noise.wav!

        echo 'generating noise profile.'

        sox !line:.wav=_noise.wav! -n noiseprof noise.prof

        echo 'cleaning original audio.'

        sox %%i !line:.wav=_clean.wav! noisered noise.prof 0.21

        echo 'cleaned audio generated.'
    )
    
)



ENDLOCAL
