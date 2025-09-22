from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import List, Optional

import logging

from PySide6 import QtCore, QtGui, QtWidgets

from ..config import Config, find_defaults_path
from ..pipeline import analyze_single_image


class DropList(QtWidgets.QListWidget):
    filesDropped = QtCore.Signal(list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setAlternatingRowColors(True)

    def dragEnterEvent(self, e: QtGui.QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e: QtGui.QDragMoveEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QtGui.QDropEvent):
        files = []
        for url in e.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.is_file():
                files.append(str(p))
        if files:
            self.filesDropped.emit(files)


class QtSignalLogHandler(logging.Handler):
    """Forward pipeline log records to a Qt signal for the log window."""

    def __init__(self, signal: QtCore.SignalInstance):
        super().__init__(level=logging.INFO)
        self._signal = signal

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
            # Include structured scoring/logging context if present.
            parts = [message]
            score = getattr(record, "score", None)
            if score is not None:
                try:
                    parts.append(f"score={float(score):.2f}")
                except Exception:
                    parts.append(f"score={score}")
            details = getattr(record, "details", None)
            if details:
                if isinstance(details, dict):
                    formatted = ", ".join(f"{k}={v}" for k, v in details.items())
                else:
                    formatted = str(details)
                parts.append(formatted)
            text = " | ".join(parts)
            self._signal.emit(text)
        except Exception:
            # Fallback to the raw message if formatting fails.
            self._signal.emit(record.getMessage())


class Worker(QtCore.QObject):
    progress = QtCore.Signal(str)
    finishedOne = QtCore.Signal(dict)
    finishedAll = QtCore.Signal()
    error = QtCore.Signal(str)

    def __init__(self, files: List[Path], cfg: Config, out_dir: Path, want_pdf: bool):
        super().__init__()
        self.files = files
        self.cfg = cfg
        self.out_dir = out_dir
        self.want_pdf = want_pdf
        self._log_handler: Optional[QtSignalLogHandler] = None

    @QtCore.Slot()
    def run(self):
        try:
            root_logger = logging.getLogger("pixspector")
            self._log_handler = QtSignalLogHandler(self.progress)
            root_logger.addHandler(self._log_handler)
            for f in self.files:
                self.progress.emit(f"Analyzing: {f.name}")
                try:
                    res = analyze_single_image(f, cfg=self.cfg, out_dir=self.out_dir, want_pdf=self.want_pdf)
                    self.finishedOne.emit(res)
                except Exception as ex:
                    self.error.emit(f"{f.name}: {ex}")
            self.finishedAll.emit()
        except Exception as e:
            self.error.emit(str(e))
            self.finishedAll.emit()
        finally:
            if self._log_handler:
                root_logger = logging.getLogger("pixspector")
                root_logger.removeHandler(self._log_handler)
                self._log_handler = None


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("pixspector")
        self.resize(1200, 760)

        self.defaults_path = find_defaults_path(Path(__file__))
        self.cfg = Config.load(self.defaults_path, None)

        # --- Widgets
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        self.listFiles = DropList()
        self.listFiles.setMinimumWidth(380)
        self.listFiles.filesDropped.connect(self.add_files)

        self.btnAdd = QtWidgets.QPushButton("Browse…")
        self.btnClear = QtWidgets.QPushButton("Clear")
        self.btnAnalyze = QtWidgets.QPushButton("Analyze")
        self.btnAnalyze.setDefault(True)

        self.chkPdf = QtWidgets.QCheckBox("Generate PDF")
        self.chkPdf.setChecked(True)

        self.outDirEdit = QtWidgets.QLineEdit(str(Path("out").resolve()))
        self.btnOutBrowse = QtWidgets.QPushButton("Choose…")

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["File", "Suspicion", "Bucket", "Open Reports"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        self.previewLabel = QtWidgets.QLabel("Preview")
        self.previewLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.previewLabel.setStyleSheet("QLabel { background: #111; color: #bbb; }")

        self.comboPreview = QtWidgets.QComboBox()
        self.comboPreview.addItems([
            "input_image.png",
            "ela_diff.png",
            "jpeg_ghosts_best_diff.png",
            "resampling_overlay.png",
            "cfa_overlay.png",
            "prnu_residual.png",
            "fft_logmag.png",
        ])

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)   # indeterminate by default
        self.progress.setVisible(False)

        # --- Layouts
        leftTop = QtWidgets.QHBoxLayout()
        leftTop.addWidget(self.btnAdd)
        leftTop.addWidget(self.btnClear)

        outRow = QtWidgets.QHBoxLayout()
        outRow.addWidget(QtWidgets.QLabel("Output folder:"))
        outRow.addWidget(self.outDirEdit, 1)
        outRow.addWidget(self.btnOutBrowse)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("Files (drag & drop):"))
        left.addLayout(leftTop)
        left.addWidget(self.listFiles, 1)
        left.addLayout(outRow)
        left.addWidget(self.chkPdf)
        left.addWidget(self.btnAnalyze)

        rightTop = QtWidgets.QHBoxLayout()
        rightTop.addWidget(QtWidgets.QLabel("Preview:"))
        rightTop.addWidget(self.comboPreview, 1)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(self.table, 2)
        right.addLayout(rightTop)
        right.addWidget(self.previewLabel, 3)
        right.addWidget(QtWidgets.QLabel("Log:"))
        right.addWidget(self.log, 1)
        right.addWidget(self.progress)

        main = QtWidgets.QHBoxLayout(central)
        main.addLayout(left, 1)
        main.addLayout(right, 2)

        # --- Signals
        self.btnAdd.clicked.connect(self.pick_files)
        self.btnClear.clicked.connect(self.clear_files)
        self.btnOutBrowse.clicked.connect(self.pick_out_dir)
        self.btnAnalyze.clicked.connect(self.on_analyze_clicked)
        self.table.cellClicked.connect(self.on_table_clicked)
        self.comboPreview.currentIndexChanged.connect(self.refresh_preview)

        # storage
        self.results_by_file: dict[str, dict] = {}

    def add_files(self, paths: List[str]):
        for p in paths:
            if not any(self.listFiles.item(i).text() == p for i in range(self.listFiles.count())):
                self.listFiles.addItem(p)

    def pick_files(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select images", "", "Images (*.jpg *.jpeg *.png *.tif *.tiff *.webp)")
        if files:
            self.add_files(files)

    def clear_files(self):
        self.listFiles.clear()
        self.table.setRowCount(0)
        self.results_by_file.clear()
        self.previewLabel.setPixmap(QtGui.QPixmap())
        self.log.clear()

    def pick_out_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder", str(Path("out").resolve()))
        if d:
            self.outDirEdit.setText(d)

    def append_log(self, msg: str):
        self.log.appendPlainText(msg)

    def on_analyze_clicked(self):
        if self.listFiles.count() == 0:
            QtWidgets.QMessageBox.information(self, "pixspector", "Add at least one image.")
            return

        out_dir = Path(self.outDirEdit.text()).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        files = [Path(self.listFiles.item(i).text()) for i in range(self.listFiles.count())]
        want_pdf = self.chkPdf.isChecked()

        self.progress.setVisible(True)
        self.log.clear()

        self.worker = Worker(files, self.cfg, out_dir, want_pdf)
        self.thread = QtCore.QThread(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.append_log)
        self.worker.finishedOne.connect(self.on_result)
        self.worker.error.connect(self.append_log)
        self.worker.finishedAll.connect(self.on_done)
        self.thread.start()

    def on_result(self, res: dict):
        file_name = Path(res["input"]).name
        self.results_by_file[file_name] = res

        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(file_name))
        self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(res.get("suspicion_index"))))
        self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(res.get("bucket_label"))))

        btn = QtWidgets.QPushButton("Open")
        def open_folder():
            folder = Path(res["artifacts_dir"]).parent  # out/<stem>_artifacts' parent is out/
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))
        btn.clicked.connect(open_folder)
        self.table.setCellWidget(row, 3, btn)

        # Auto-preview the first result
        if self.table.rowCount() == 1:
            self.refresh_preview()

    def on_done(self):
        self.append_log("Done.")
        self.progress.setVisible(False)
        self.thread.quit()
        self.thread.wait()

    def current_result(self) -> Optional[dict]:
        row = self.table.currentRow()
        if row < 0:
            return None
        fname = self.table.item(row, 0).text()
        return self.results_by_file.get(fname)

    def on_table_clicked(self, row: int, col: int):
        self.refresh_preview()

    def refresh_preview(self):
        res = self.current_result()
        if not res:
            self.previewLabel.setPixmap(QtGui.QPixmap())
            return
        artifacts = Path(res["artifacts_dir"])
        sel = self.comboPreview.currentText()
        img_path = artifacts / sel
        if not img_path.exists():
            self.previewLabel.setText(f"{sel} not found")
            return
        pix = QtGui.QPixmap(str(img_path))
        self.previewLabel.setPixmap(pix.scaled(self.previewLabel.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
