#include "drampool_daemon.h"

int main(int argc, char** argv)
{
    UC::DramPool::DramPoolDaemon daemon;
    return daemon.Run(argc, argv);
}
