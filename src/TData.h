/************************************************/
//TData.h
/************************************************/
#ifndef TDATA_H
#define TDATA_H

#include <string>
#include <fstream>
#include <list>

///Чтение строки данных из файла
inline bool ReadStringFromFile(const std::string &file_name, std::string &s)
{
	bool result;
	s.clear();
  std::ifstream in_stream;
  in_stream.open(file_name.c_str());
	result = in_stream.is_open();
	if (result)
	{
		in_stream.seekg(0);
		std::string temp;
		while (std::getline(in_stream, temp))
			s += temp + "\n";	
		in_stream.close();
	}
  return result;
};

///Запись строки в файл
inline bool WriteStringToFile(const std::string &file_name, const std::string &s)
{
	bool result;
  std::filebuf of_buffer;
	std::ostream out_stream(&of_buffer);
	of_buffer.open(file_name, std::ios::out|std::ios::trunc);					//Запись	
	result = of_buffer.is_open();
	if (result)
	{	   
		out_stream<<s;							 	
		out_stream.clear();
		of_buffer.close();
	}
	return result;
};

/************************************************/

#ifdef _WIN32

///Рекурсивный поиск файлов во вложенной структуре подкаталогов
class TDataLoader : public std::list <std::string> {
protected:
	///Возвращает список всех подкаталогов
	void GetSubDirs( const char * PDataDirS, std::list <std::string> * PDataDirList );
	///Формирует список файлов, хранящихся во всех подкаталогах списка
	void GetFileList( const std::list <std::string> * PDataDirVector, const char * PFileMask );			
public:
	TDataLoader( const char * PDataDir, const char * PFileMask, const bool SearchSubDirs );
	~TDataLoader() {};
};

///Рекурсивный поиск файлов во вложенной структуре подкаталогов
std::list<std::string> getFilesNames(std::string dirname, std::string mask);

#endif

/************************************************/

#endif
