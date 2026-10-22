"""
Automatic Versioning & Design Modification Archiving Engine for 2D Optics.
Automatically creates timestamped backup copies and version records on every Save / Save As.
Allows tracking, comparing, and restoring historical design revisions from disk archives.
"""
import os
import json
import shutil
from datetime import datetime
from typing import Optional, Any, Dict, List


class VersionManager:
    """
    Manages automatic on-save archiving and historical version copies of optical designs.
    """

    @staticmethod
    def archive_version(
        file_path: str,
        stack: Any,
        system_config: Optional[Any] = None,
        wavelengths: Optional[List[float]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        analysis_config: Optional[Dict[str, Any]] = None,
        description: str = ""
    ) -> str:
        """
        Creates a timestamped copy of the entire optical design in a .history directory.
        Returns the path of the created archive file.
        """
        if not file_path:
            return ""

        try:
            abs_path = os.path.abspath(file_path)
            proj_dir = os.path.dirname(abs_path)
            base_filename = os.path.basename(abs_path)
            stem, ext = os.path.splitext(base_filename)
            if not ext:
                ext = ".json"

            # 1. Create .history directory inside the project directory
            history_dir = os.path.join(proj_dir, ".history")
            os.makedirs(history_dir, exist_ok=True)

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamp_readable = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 2. Count existing versions from index file
            history_index_file = os.path.join(history_dir, f"{stem}_history_index.json")
            version_records = []
            if os.path.exists(history_index_file):
                try:
                    with open(history_index_file, "r", encoding="utf-8") as f:
                        version_records = json.load(f)
                except Exception:
                    version_records = []

            version_num = len(version_records) + 1
            archive_filename = f"{stem}_v{version_num:03d}_{timestamp_str}{ext}"
            archive_path = os.path.join(history_dir, archive_filename)

            # 3. Save copy of the complete system to the archive file
            if hasattr(stack, "save_to_json"):
                stack.save_to_json(
                    filepath=archive_path,
                    wavelengths=wavelengths,
                    system_config=system_config,
                    optimizer_config=optimizer_config,
                    analysis_config=analysis_config
                )
            elif os.path.exists(abs_path):
                shutil.copy2(abs_path, archive_path)

            # 4. Update history index file
            surf_cnt = len(stack.surfaces) if hasattr(stack, "surfaces") else 0
            file_sz = os.path.getsize(archive_path) if os.path.exists(archive_path) else 0
            
            record = {
                "version": version_num,
                "filename": archive_filename,
                "path": archive_path,
                "timestamp": timestamp_readable,
                "description": description or f"Saved Version #{version_num}",
                "surfaces_count": surf_cnt,
                "filesize_bytes": file_sz
            }
            version_records.append(record)

            with open(history_index_file, "w", encoding="utf-8") as f:
                json.dump(version_records, f, indent=2)

            return archive_path
        except Exception as e:
            print(f"[VersionManager] Warning: failed to archive version: {e}")
            return ""

    @staticmethod
    def get_archived_versions(file_path: str) -> List[Dict[str, Any]]:
        """
        Retrieves list of all saved version archives for a given project file.
        """
        if not file_path:
            return []
        try:
            abs_path = os.path.abspath(file_path)
            proj_dir = os.path.dirname(abs_path)
            base_filename = os.path.basename(abs_path)
            stem, _ = os.path.splitext(base_filename)
            history_index_file = os.path.join(proj_dir, ".history", f"{stem}_history_index.json")
            if os.path.exists(history_index_file):
                with open(history_index_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return []
