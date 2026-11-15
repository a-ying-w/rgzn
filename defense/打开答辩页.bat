@echo off
rem 双击即可全屏打开答辩页面（Chrome / Edge）
set "DECK=%~dp0defense.html"
if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" (
  start "" "%ProgramFiles%\Google\Chrome\Application\chrome.exe" --start-fullscreen "%DECK%"
) else if exist "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe" (
  start "" "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe" --start-fullscreen "%DECK%"
) else (
  start "" "%DECK%"
)
