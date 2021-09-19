#ifndef NODE_CPP
#define NODE_CPP
#include "DevTask.h"
template<class T>
    class Node
    {
    public:
        Node();
	~Node();
        T * value;
        Node<T> * next;
    };

    template<class T>
    Node<T>::Node()
    {
	value=NULL;
        next = NULL;
    }
    template<class T>
    Node<T>::~Node()
    {
    }
#endif
