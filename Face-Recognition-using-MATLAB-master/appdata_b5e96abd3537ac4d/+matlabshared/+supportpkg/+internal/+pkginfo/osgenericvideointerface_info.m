function supportpkg = osgenericvideointerface_info()
%OSGENERICVIDEOINTERFACE_INFO Return Image Acquisition Toolbox Support Package for OS Generic Video Interface information.

%   Copyright 2020 The MathWorks, Inc.

supportpkg = hwconnectinstaller.SupportPackage();
supportpkg.Name          = 'OS Generic Video Interface';
supportpkg.Version       = '15.2.0';
supportpkg.Platform      = 'PCWIN,PCWIN64,GLNXA64,MACI64';
supportpkg.Visible       = '1';
supportpkg.FwUpdate      = '';
supportpkg.Url           = 'http://www.mathworks.com';
supportpkg.DownloadUrl   = '';
supportpkg.LicenseUrl    = '';
supportpkg.BaseProduct   = 'Image Acquisition Toolbox';
supportpkg.AllowDownloadWithoutInstall = true;
supportpkg.FullName      = 'Image Acquisition Toolbox Support Package for OS Generic Video Interface';
supportpkg.DisplayName      = 'OS Generic Video Interface';
supportpkg.SupportCategory      = 'hardware';
supportpkg.CustomLicense = '';
supportpkg.CustomLicenseNotes = '';
supportpkg.ShowSPLicense = true;
supportpkg.Folder        = 'genericvideo';
supportpkg.Release       = '(R2015b)';
supportpkg.DownloadDir   = 'D:\Face-Recognition-using-MATLAB-master\downloads\osgenericvideointerface_download';
supportpkg.InstallDir    = 'D:\Face-Recognition-using-MATLAB-master';
supportpkg.IsDownloaded  = 0;
supportpkg.IsInstalled   = 1;
supportpkg.RootDir       = 'D:\Face-Recognition-using-MATLAB-master\osgenericvideointerface\toolbox\imaq\supportpackages\genericvideo';
supportpkg.DemoXml       = '';
supportpkg.ExtraInfoCheckBoxDescription       = '';
supportpkg.ExtraInfoCheckBoxCmd       = '';
supportpkg.FwUpdate      = '';
supportpkg.PreUninstallCmd      = 'matlab%3Aimaqreset%3B';
supportpkg.InfoUrl      = '';
supportpkg.BaseCode     = 'OSVIDEO';
supportpkg.SupportTypeQualifier      = 'Standard';
supportpkg.CustomMWLicenseFiles      = '';
supportpkg.InstalledDate      = '02-Aug-2020 14:17:17';
supportpkg.InfoText      = 'Acquire+video+and+images+from+generic+video+capture+devices.';
supportpkg.Path{1}      = 'D:\Face-Recognition-using-MATLAB-master\osgenericvideointerface\toolbox\imaq\supportpackages\genericvideo';

% Third party software information
