' ==============================================================================
' Script chay ngam VBxIris Backend (Khong hien cua so CMD/Terminal)
' ==============================================================================
Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Lay duong dan thu muc hien tai
CurrentDir = fso.GetParentFolderName(WScript.ScriptFullName)

' Duong dan file thuc thi Python va file app.py
PythonExe = CurrentDir & "\backend\venv\Scripts\python.exe"
AppScript = CurrentDir & "\backend\app.py"
BackendDir = CurrentDir & "\backend"

' Kiem tra file ton tai
If Not fso.FileExists(PythonExe) Then
    PythonExe = "python"
End If

' Chay ngam an hoan toan (Tham so 0 = An cua so CMD, False = Khong cho ket thuc)
Command = """" & PythonExe & """ """ & AppScript & """"
WshShell.CurrentDirectory = BackendDir
WshShell.Run Command, 0, False
