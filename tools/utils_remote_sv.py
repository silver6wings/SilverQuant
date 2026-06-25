"""远程选股列表服务：push / pull stock codes。"""

import datetime
from typing import Optional

import requests


def build_stock_list_key(prefix: str, date: datetime.date | None = None) -> str:
    day = date or datetime.datetime.now().date()
    return f'{prefix}_{day.strftime("%Y%m%d")}'


def push_stock_codes(
    codes: list[str],
    key: str,
    host: str,
    auth: str,
    *,
    timeout: int = 10,
) -> tuple[bool, str]:
    """推送选股列表到指定 key。成功返回 (True, '')，失败返回 (False, error_msg)。"""
    try:
        response = requests.post(
            f'{host}/stocks/push/{key}',
            params={'auth': auth},
            json={'value': codes},
            timeout=timeout,
        )
    except requests.RequestException as e:
        return False, str(e)

    if response.status_code == 200:
        return True, ''
    if response.status_code == 404:
        try:
            response_json = response.json()
        except ValueError:
            return False, 'Not Found'
        return False, response_json.get('error', 'Not Found')
    return False, f'HTTP {response.status_code}'


def pull_stock_codes(
    key: str,
    host: str,
    auth: str,
    *,
    timeout: int = 10,
) -> tuple[Optional[list[str]], str]:
    """从指定 key 拉取选股列表。成功返回 (codes, '')，失败返回 (None, error_msg)。"""
    try:
        response = requests.get(
            f'{host}/stocks/pull/{key}',
            params={'auth': auth},
            timeout=timeout,
        )
    except requests.RequestException as e:
        return None, str(e)

    if response.status_code == 200:
        return response.json(), ''
    if response.status_code == 404:
        try:
            response_json = response.json()
        except ValueError:
            return None, 'Not Found'
        return None, response_json.get('error', 'Not Found')
    return None, f'HTTP {response.status_code}'


def push_stock_today_codes(
    codes: list[str],
    prefix: str,
    host: str,
    auth: str,
    *,
    timeout: int = 10,
) -> tuple[bool, str]:
    """推送当日选股列表。成功返回 (True, '')，失败返回 (False, error_msg)。"""
    return push_stock_codes(codes, build_stock_list_key(prefix), host, auth, timeout=timeout)


def pull_stock_today_codes(
    prefix: str,
    host: str,
    auth: str,
    *,
    timeout: int = 10,
) -> tuple[Optional[list[str]], str]:
    """拉取当日选股列表。成功返回 (codes, '')，失败返回 (None, error_msg)。"""
    return pull_stock_codes(build_stock_list_key(prefix), host, auth, timeout=timeout)
