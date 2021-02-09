#ifndef TINDEXED_OBJECTS_H
#define TINDEXED_OBJECTS_H

#include <vector>
#include <map>

///Упорядоченное хранение объектов с быстрым доступом по индексу и проверкой уникальности
///!!!По памяти не оптимально - объекты в двух экземплярах
template <class T> 
class TIndexedObjects {
protected:
	///Объект по номеру найти легко, т.к. достаточно просто поднять элемент массива с индексом == номеру
	std::vector <T> IndexedObjects;
	///Номер по строке тоже найти легко, т.к. заведено следующее отображение:
	std::map<T, unsigned long > ObjectIndices;
public:
	unsigned long Size(){return (unsigned long)IndexedObjects.size();};

	///Вернуть объект по индексу
	T GetObjByIdx(const unsigned long idx)
	{
		if ((idx < 0) || (idx > (unsigned long)IndexedObjects.size())) return T();
		else return IndexedObjects[idx];
	};

	///Вернуть индекс по объекту
	///Добавит новый элемент, если нет такого и присвоит ему label, если add_new_if_not_exists == true
	unsigned long GetIdxByObj(const T obj, const bool add_new_if_not_exists = true)
	{
		std::map<T, unsigned long >::iterator it = ObjectIndices.find(obj);
		if (it != ObjectIndices.end())
		{
			return it->second;
		} else {
			if (add_new_if_not_exists)
			{
				ObjectIndices.insert(std::make_pair(obj, (unsigned long)IndexedObjects.size()));
				IndexedObjects.push_back(obj);
				return (unsigned long)IndexedObjects.size() - 1;
			}
		};
		return -1;
	};

	void Clear()
	{
		IndexedObjects.clear();
		ObjectIndices.clear();
	};
};

#endif