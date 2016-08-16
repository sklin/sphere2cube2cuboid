#include "MyServer.h"
#include "MessageQueue.h"
#include "ImageProcessor.h"
#include "mHeader.hpp"
//#include "lz4.h"

#include <pthread.h>
#include <unistd.h> // for usleep()

pthread_mutex_t lock;
MessageQueue messageQueue;

/////////////////////////////////////////////
// Main
/////////////////////////////////////////////

void* Thread_Socket(void* ptr)
{
    MyServer server(5566);
    server.Prepare();

    // MainLoop
    //server.MainLoop();
    while(true) {
        // accept
        server.Accept();
        while(true) {
            char buf[1024];
            // receive from client
            int nrecv = server.Recv(buf, 1024);
            if(nrecv <= 0) {
                server.CloseClient();
                break;
            }
            std::cout << "Recv: " << nrecv << " byte" << std::endl;

            // lock
            pthread_mutex_lock(&lock);

            // response to client
            if(messageQueue.Size() > 0 ) {
                char *sndbuf = messageQueue.front()->buffer;
                int sendLength = messageQueue.front()->length;
                int nsend = server.Send(sndbuf, sendLength);
                //int nsend = server.Send(sndbuf, 1024);
                messageQueue.Pop();
            }
            else {
                // current no message
                const char *sndbuf = "Current no message\n";
                int nsend = server.Send(sndbuf, strlen(sndbuf));
            }

            // unlock
            pthread_mutex_unlock(&lock);
        }
    }
    return NULL;
}

void* Thread_Process(void* ptr)
{
    ImageProcessor imageProcessor;
    while(true) {
        // process image
        int returnSize = 0;
        char *result = imageProcessor.process(returnSize);

        //// lz4 compress
        //char buffer[10240];
        //int compressSize = 0;
        //compressSize = LZ4_compress_default(result, buffer, strlen(result), 10240);
        //std::cout << "compressSize: " << compressSize << std::endl;

        // lock
        pthread_mutex_lock(&lock);

        // push into queue
        if(messageQueue.Size() == MQ_SIZE_LIMIT) {
            messageQueue.Pop();
        }
        messageQueue.Push(result, returnSize);

        // unlock
        pthread_mutex_unlock(&lock);

        //usleep(1000000);
    }

    return NULL;
}

int main(int argc, char **argv)
{
    /*
    test_sphere2cube2cuboid(1024);
    */
    pthread_t tid[2];
    void *ret;

    if (pthread_mutex_init(&lock, NULL) != 0) {
        std::cerr << "mutex init failed" << std::endl;
        return 0;
    }

    pthread_create(&tid[0], NULL, Thread_Socket, NULL);
    pthread_create(&tid[1], NULL, Thread_Process, NULL);
    pthread_join(tid[0], &ret);
    pthread_join(tid[1], &ret);

    return 0;
}

