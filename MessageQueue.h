#ifndef MESSAGE_QUEUE_H
#define MESSAGE_QUEUE_H

#include <iostream>
#include <queue>

#define MQ_SIZE_LIMIT 10

struct PacketInfo {
    char *buffer;
    int length;

    PacketInfo(char* _buffer = NULL, int _length = 0): buffer(_buffer), length(_length) {}
    ~PacketInfo() {
        //if(_buffer) delete [] buffer; // TODO: Maybe it should be done outside
    }
};

class MessageQueue {
public:
    void Push(char *buf, int length)
    {
        mQueue.push(new PacketInfo(buf, length));
        std::cout << "Push(), current size is " << mQueue.size() << std::endl;
    }

    void Pop() // need to free the memory
    {
        PacketInfo *front = mQueue.front();
        delete [] front->buffer;
        delete front;
        mQueue.pop();
        std::cout << "Pop(), current size is " << mQueue.size() << std::endl;
    }

    unsigned Size() { return mQueue.size(); }

    PacketInfo* front() { return mQueue.front(); }
private:
    std::queue<PacketInfo*> mQueue;
};

#endif
