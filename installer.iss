; Сценарий Inno Setup: установщик программной системы «РегПрогноз»
; Собирается из build.bat (или вручную: открыть файл в Inno Setup → Build → Compile)

#define AppName "РегПрогноз"
#define AppVersion "1.0"
#define AppExe "RegPrognoz.exe"

[Setup]
AppId={{6E2B6C1A-5B7E-4C55-9C1E-2B7A0D4E8F31}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=А. О. Кортова, МГТУ им. Н.Э. Баумана, ИУ6-55Б
DefaultDirName={autopf}\RegPrognoz
DefaultGroupName={#AppName}
; установка без прав администратора (в папку пользователя)
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
; только 64-разрядные Windows 10/11 (п. 4.5.1 ТЗ)
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir=installer
OutputBaseFilename=RegPrognoz_Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName={#AppName}

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "Создать значок на рабочем столе"; GroupDescription: "Дополнительно:"

[Files]
; всё содержимое папки, собранной PyInstaller
Source: "dist\RegPrognoz\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\Удалить {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Запустить {#AppName}"; Flags: nowait postinstall skipifsilent
