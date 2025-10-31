"""
BXC_VideoNet 版本信息
作者：北小菜
"""

# 当前版本号
CUR_VERSION = "1.01"

# 版本信息
VERSION_INFO = {
    'version': CUR_VERSION,
    'name': 'BXC_VideoNet',
    'author': '北小菜',
    'description': '视频动作识别深度学习框架',
    'release_date': '2025-10-31'
}


def get_version_string():
    """获取完整版本字符串"""
    return f"{VERSION_INFO['name']} v{VERSION_INFO['version']}"


def print_version():
    """打印版本信息"""
    print("=" * 60)
    print(f"{VERSION_INFO['name']} - {VERSION_INFO['description']}")
    print(f"版本: v{VERSION_INFO['version']}")
    print(f"作者: {VERSION_INFO['author']}")
    print(f"发布日期: {VERSION_INFO['release_date']}")
    print("=" * 60)
