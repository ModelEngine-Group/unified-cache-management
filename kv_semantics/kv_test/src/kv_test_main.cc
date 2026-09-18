#include "kv_test_app.h"

int main(int argc, char** argv)
{
    kv::bench::KvTestApp app;
    return app.Run(argc, argv);
}
