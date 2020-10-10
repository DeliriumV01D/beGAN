#ifndef TINDEXED_OBJECTS_H
#define TINDEXED_OBJECTS_H

#include <vector>
#include <map>

///������������� �������� �������� � ������� �������� �� ������� � ��������� ������������
///!!!�� ������ �� ���������� - ������� � ���� �����������
template <class T> 
class TIndexedObjects {
protected:
	///������ �� ������ ����� �����, �.�. ���������� ������ ������� ������� ������� � �������� == ������
	std::vector <T> IndexedObjects;
	///����� �� ������ ���� ����� �����, �.�. �������� ��������� �����������:
	std::map<T, unsigned long > ObjectIndices;
public:
	unsigned long Size(){return (unsigned long)IndexedObjects.size();};

	///������� ������ �� �������
	T GetObjByIdx(const unsigned long idx)
	{
		if ((idx < 0) || (idx > (unsigned long)IndexedObjects.size())) return T();
		else return IndexedObjects[idx];
	};

	///������� ������ �� �������
	///������� ����� �������, ���� ��� ������ � �������� ��� label, ���� add_new_if_not_exists == true
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