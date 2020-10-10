/************************************************/
//TData.cpp
/************************************************/
#include "TData.h"

#include <io.h>
#include <iostream>

#ifdef _WIN32
TDataLoader :: TDataLoader( const char * PDataDir, const char * PFileMask, const bool SearchSubDirs )
{
	std::list <std::string> DataDirList;
	std::string SDataDir (PDataDir);
	if (SearchSubDirs)
	{
		GetSubDirs( PDataDir, & DataDirList );
		GetFileList( & DataDirList, PFileMask );
	} else 
	{
		DataDirList.push_back(SDataDir);
		GetFileList( & DataDirList, PFileMask );
	};
};

//Возвращает список всех подкаталогов
void TDataLoader :: GetSubDirs( const char * PDataDirS, std::list <std::string> * PDataDirList ) 
{
	std::string SDataDir(PDataDirS);
	std::string DirMask = SDataDir;
	std::string sDir;
  std::string RootDir = PDataDirS; RootDir+="/..";
	std::string Root	=	PDataDirS; Root+="/.";
	DirMask+="/*.*";
	struct _finddata_t _SearchRec;
	intptr_t hfile;

	PDataDirList->push_back(SDataDir);
	
	_SearchRec.attrib = 0x00000010; //Directory
	hfile = _findfirst(DirMask.data(), &_SearchRec);
	if (hfile != -1)
	{	
		do{
			sDir.clear();
			sDir+=SDataDir;
			sDir+="/";
			sDir+=_SearchRec.name;
			if ((sDir != Root) && (sDir != RootDir))
				PDataDirList->push_back(sDir);
		} while (_findnext(hfile, &_SearchRec) != -1);
	}
	_findclose(hfile);
};	// of GetSubDirs

//Формирует список файлов, хранящихся во всех подкаталогах списка
void TDataLoader :: GetFileList( const std::list <std::string> * PDataDirList, const char * PFileMask ) 
{
	struct _finddata_t _SearchRec;
	std::string _DataFilesMask;
	std::string RootDir = *(PDataDirList->begin()); RootDir+="/..";
	std::string Root	=	*(PDataDirList->begin()); Root+="/.";
	std::string sFile;
	std::list <std::string> :: const_iterator CurrentListPosition;
	intptr_t hfile;
	_SearchRec.attrib = 0; //AnyFile

	for ( CurrentListPosition = PDataDirList->begin(); CurrentListPosition != PDataDirList->end(); ++CurrentListPosition )
	{	
			_DataFilesMask = (*CurrentListPosition);
			_DataFilesMask += "/";
			_DataFilesMask += PFileMask;//"/*.sin";
			hfile = _findfirst(_DataFilesMask.data(), &_SearchRec);
			if (hfile != -1)
				do{
					sFile.clear();
					sFile = (*CurrentListPosition);
					sFile += "/";
					sFile += _SearchRec.name;
					if ((sFile != Root) && (sFile != RootDir))
						this->push_back(sFile);
				}while(_findnext(hfile, &_SearchRec) != -1);
			_findclose(hfile);
	}
}		// of GetFileList

///Рекурсивный поиск файлов во вложенной структуре подкаталогов
std::list<std::string> getFilesNames(std::string dirname, std::string mask)
{
	struct _finddata_t fd, fdsd;
	std::string findPath = dirname + "\\" + mask;
	std::string dirPath = dirname + "\\" + "*";
	std::list<std::string> res;

	//Перебор файлов в директории
	intptr_t hFile = _findfirst(findPath.c_str(), &fd);
	fd.attrib = 0; //AnyFile
	if (hFile != -1L)
		do
		{
			if (!(fd.attrib & _A_SUBDIR))
				res.push_back(dirname + "\\" + fd.name);
		} while (_findnext(hFile, &fd) == 0);
	_findclose(hFile);

	//перебор поддиректорий
	fdsd.attrib = _A_SUBDIR; //Directory
	hFile = _findfirst(dirPath.c_str(), &fdsd);
	if (hFile != -1L)
		do
		{
			if ((fdsd.attrib & _A_SUBDIR) && fdsd.name != std::string(".") && fdsd.name != std::string(".."))
			{
				res.splice(res.end(), getFilesNames(dirname + "\\" + fdsd.name, mask));
			}
		} while (_findnext(hFile, &fdsd) == 0);
	_findclose(hFile);

	return res;
}

#endif //_WIN32