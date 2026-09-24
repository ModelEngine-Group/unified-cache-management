# 版权所有（c）华为技术有限公司 2012-2026
from setuptools import setup,Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "kvb_agent_connector",                 #模块名（import时使用的名字）
        ["kvb_agent_connector.py"],            #源文件（可写多个）
        language_level=3,                      #Python 3 语法
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],  # 可选：屏蔽numpy警告
        annotation_typing=False,
    ),
    Extension(
        "lcp_session",                         #模块名(import时使用的名字)
        ["lcp_session.py"],                    #源文件（可写多个）
        language_level=3,                      #Python 3 语法
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],  # 可选：屏蔽numpy警告
        annotation_typing=False,
    ),
    Extension(
        "get_sparse_block_table",                 #模块名（import时使用的名字）
        ["get_sparse_block_table.py"],            #源文件（可写多个）
        language_level=3,                      #Python 3 语法
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],  # 可选：屏蔽numpy警告
        annotation_typing=False,
    ),
    Extension(
        "lightweight_cache",                 #模块名（import时使用的名字）
        ["kv_bridge_connector.py"],            #源文件（可写多个）
        language_level=3,                      #Python 3 语法
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],  # 可选：屏蔽numpy警告
        annotation_typing=False,
    ),
    Extension(
        "tool_call_pruning",                 #模块名（import时使用的名字）
        ["tool_call_pruning.py"],            #源文件（可写多个）
        language_level=3,                      #Python 3 语法
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],  # 可选：屏蔽numpy警告
        annotation_typing=False,
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "boundscheck": False,          # 关闭数组越界检查（提速）
            "wraparound": False,           # 关闭负数索引支持（提速）
            "binding": True,              
        },
        annotate=True,                     # 生成.html 查看C转换质量
    ),
    zip_safe = False
)
