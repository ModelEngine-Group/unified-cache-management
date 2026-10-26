file(GLOB_RECURSE UCM_PROXY_CORE_SOURCE_FILES CONFIGURE_DEPENDS "*.cc")
list(FILTER UCM_PROXY_CORE_SOURCE_FILES EXCLUDE REGEX "proxy\\.py\\.cc$")

# Generate the built-in store table from the ucm_add_store registrations.
# Entries are "name=$<TARGET_FILE:...>"; the generator expressions expand to
# the plugins' absolute build paths at generation time. A new built-in store
# therefore needs no proxy-side change.
get_property(UCM_BUILTIN_STORE_ENTRIES GLOBAL PROPERTY UCM_BUILTIN_STORES)
set(UCM_BUILTIN_TABLE_CONTENT "#include <string>\n#include <utility>\n#include <vector>\n")
string(APPEND UCM_BUILTIN_TABLE_CONTENT "namespace UC::Proxy {\n")
string(APPEND UCM_BUILTIN_TABLE_CONTENT "const std::vector<std::pair<std::string, std::string>>& BuiltinStoreTable()\n{\n")
string(APPEND UCM_BUILTIN_TABLE_CONTENT "    static const std::vector<std::pair<std::string, std::string>> table = {\n")
foreach(UCM_BUILTIN_ENTRY IN LISTS UCM_BUILTIN_STORE_ENTRIES)
    string(REGEX MATCH "^([^=]+)=(.*)$" UCM_BUILTIN_MATCH "${UCM_BUILTIN_ENTRY}")
    if(NOT CMAKE_MATCH_1)
        message(FATAL_ERROR "malformed built-in store entry '${UCM_BUILTIN_ENTRY}'")
    endif()
    string(APPEND UCM_BUILTIN_TABLE_CONTENT "        {\"${CMAKE_MATCH_1}\", \"${CMAKE_MATCH_2}\"},\n")
endforeach()
string(APPEND UCM_BUILTIN_TABLE_CONTENT "    };\n    return table;\n}\n")
string(APPEND UCM_BUILTIN_TABLE_CONTENT "}  // namespace UC::Proxy\n")
file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/builtin_stores.cc CONTENT "${UCM_BUILTIN_TABLE_CONTENT}")

add_library(ucm_proxy_core OBJECT ${UCM_PROXY_CORE_SOURCE_FILES} ${CMAKE_CURRENT_BINARY_DIR}/builtin_stores.cc)
target_include_directories(ucm_proxy_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ucm_proxy_core PUBLIC ucm_build_options ucm_common ucm_store ${CMAKE_DL_LIBS})

pybind11_add_module(ucm_proxy proxy.py.cc)
target_link_libraries(ucm_proxy PRIVATE ucm_build_options ucm_common ucm_store ucm_proxy_core)
