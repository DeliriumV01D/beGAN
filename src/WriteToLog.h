#ifndef WRITE_TO_LOG_H
#define WRITE_TO_LOG_H

#include <string>
#include <exception>

///TException
/**
Нужно установить одну директиву из SHOW_USING_QT_DIALOG, SHOW_USING_CERR, WRITE_LOG_IN_FILE

Пример использования:
@code
throw TException("Exception text")

try {
...
} catch (exception &e) {
  TException * E = dynamic_cast<TException *>(&e);
  if (E)
    std::cout<<E->what()<<std::endl;
  else
    std::cout<<e.what()<<std::endl;
if (E) throw (*E);
else throw e;
};
@endcode

Основная концепция исключений с++: "деструкторы локальных объектов вызываются всегда,
независимо от способа возврата из функции (с помощью return или в связи с выбросом исключения)" */
class TException: public std::exception {
protected:
  std::string Text;
public:
  TException(const std::string &text)
  {
	  Text = text;
  };
  
  ~TException() throw(){};

  virtual const char* what()
  {
    return Text.c_str();
  }
};

//Отображение текста исключения с использованием диалога
#ifdef SHOW_USING_QT_DIALOG
#include <QObject>
#include <QMessageBox>
#endif

//Вывод текста исключения в поток
#ifdef SHOW_USING_CERR
#include <iostream>
using std::cerr;
#endif

//Запись в лог-файл
#ifdef WRITE_LOG_TO_FILE
#endif

#include <stdexcept>
#include <string>

inline void WriteToLog(const char * text){	
	//Отображение текста исключения с использованием диалога
	#ifdef SHOW_USING_QT_DIALOG
	QMessageBox::critical(	0, QObject::tr("Ошибка"),	//warning
													QObject::tr(text),
													QMessageBox::Ok);					//QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel);
	#endif

	//Выводим текст исключения в поток
	#ifdef SHOW_USING_CERR
	cerr<<text<<std::endl;
	#endif

	//Записываем в лог-файл
	#ifdef WRITE_LOG_IN_FILE
	#endif
};

#endif// WRITE_TO_LOG_H
