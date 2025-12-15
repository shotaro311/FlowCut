"""システム通知を送信するユーティリティ。"""
from __future__ import annotations

import logging
import platform
import subprocess

logger = logging.getLogger(__name__)


def send_notification(title: str, message: str) -> bool:
    """システム通知を送信する。

    Args:
        title: 通知タイトル
        message: 通知メッセージ

    Returns:
        通知が正常に送信されたかどうか
    """
    system = platform.system()

    try:
        if system == "Darwin":  # macOS
            return _send_macos_notification(title, message)
        elif system == "Windows":
            return _send_windows_notification(title, message)
        else:
            logger.warning("未対応のOS: %s", system)
            return False
    except Exception as exc:
        logger.warning("通知送信に失敗しました: %s", exc)
        return False


def _send_macos_notification(title: str, message: str) -> bool:
    """macOSのシステム通知を送信する。"""
    # AppleScriptでDisplay Notificationを使用
    script = f'''
    display notification "{message}" with title "{title}"
    '''
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        logger.info("通知送信完了: %s", title)
        return True
    else:
        logger.warning("osascript失敗: %s", result.stderr)
        return False


def _send_windows_notification(title: str, message: str) -> bool:
    """Windowsのトースト通知を送信する。"""
    try:
        # PowerShellでトースト通知を送信
        script = f'''
        [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
        [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
        $template = "<toast><visual><binding template='ToastText02'><text id='1'>{title}</text><text id='2'>{message}</text></binding></visual></toast>"
        $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
        $xml.LoadXml($template)
        $toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
        [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("FlowCut").Show($toast)
        '''
        result = subprocess.run(
            ["powershell", "-Command", script],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            logger.info("通知送信完了: %s", title)
            return True
        else:
            logger.warning("PowerShell通知失敗: %s", result.stderr)
            return False
    except Exception as exc:
        logger.warning("Windows通知送信に失敗: %s", exc)
        return False


__all__ = ["send_notification"]
