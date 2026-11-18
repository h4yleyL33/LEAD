' ---------------------------------------------------------------------------
' AIoT Dashboard launcher
' Starts the FastAPI service in the background, waits for /health, then opens
' the dashboard in the default browser. start_web.ps1 detaches uvicorn, so the
' server keeps running after this script exits; stop it with the desktop
' shortcut "AIoT - Stop Server".
' ---------------------------------------------------------------------------
Option Explicit

Dim shell, fso, root, cmd, rc, base, tries, ok, exec

Set shell = CreateObject("WScript.Shell")
Set fso   = CreateObject("Scripting.FileSystemObject")

' Resolve the repo root: use the shortcut's working directory when it really
' points at the repo, otherwise fall back to the clone location.
root = shell.CurrentDirectory
If Not fso.FileExists(fso.BuildPath(root, "scripts\start_web.ps1")) Then
    root = "E:\project\LAB_AI\AIoT"
End If

base = "http://127.0.0.1:8000"
cmd  = "powershell -NoProfile -ExecutionPolicy Bypass -File """ & _
       fso.BuildPath(root, "scripts\start_web.ps1") & """ -Port 8000 -NoBrowser"

' 0 = hidden window, False = do not wait.
rc = shell.Run(cmd, 0, False)
If rc <> 0 Then
    MsgBox "Gagal menjalankan server." & vbCrLf & vbCrLf & _
           "Jalankan manual untuk melihat errornya:" & vbCrLf & _
           "  powershell -ExecutionPolicy Bypass -File scripts\start_web.ps1" & vbCrLf & vbCrLf & _
           "Log: " & fso.BuildPath(root, "logs\web_server.log"), _
           vbExclamation, "AIoT Dashboard"
    WScript.Quit 1
End If

' Poll /health with curl.exe (present on Windows 10+) rather than MSXML2.XMLHTTP,
' which is not reliably registered on every host.
ok = False
For tries = 1 To 40
    Set exec = shell.Exec("curl.exe --connect-timeout 1 --max-time 2 -s -o NUL -w ""%{http_code}"" " & base & "/health")
    Do While exec.Status = 0
        WScript.Sleep 50
    Loop
    If Trim(exec.StdOut.ReadAll) = "200" Then
        ok = True
        Exit For
    End If
    WScript.Sleep 500
Next

If Not ok Then
    MsgBox "Server tidak menjawab " & base & "/health." & vbCrLf & vbCrLf & _
           "Lihat log:" & vbCrLf & _
           fso.BuildPath(root, "logs\web_server.log") & vbCrLf & _
           fso.BuildPath(root, "logs\web_server.err.log"), _
           vbExclamation, "AIoT Dashboard"
    WScript.Quit 1
End If

shell.Run """" & base & "/"""