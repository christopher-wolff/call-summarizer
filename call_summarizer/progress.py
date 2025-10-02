#!/usr/bin/env python3
"""Progress tracking for the call summarizer pipeline."""

import threading
from pathlib import Path
from typing import Dict, Optional, Any

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn, TaskID
from rich.table import Table


class PipelineProgress:
    """Manages progress tracking for all pipeline stages."""
    
    def __init__(self):
        self.console = Console()
        self.layout = Layout()
        
        # Create progress bars for each stage with better configuration
        self.audio_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(style="blue"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
            expand=True,
            refresh_per_second=10
        )
        
        self.transcription_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(style="blue"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
            expand=True,
            refresh_per_second=10
        )
        
        self.summarization_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(style="blue"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
            expand=True,
            refresh_per_second=10
        )
        
        # Task tracking
        self.audio_tasks: Dict[str, TaskID] = {}
        self.transcription_tasks: Dict[str, TaskID] = {}
        self.summarization_tasks: Dict[str, TaskID] = {}
        
        # Layout setup
        self._setup_layout()
        
        # Live display
        self.live: Optional[Live] = None
        self._lock = threading.Lock()
    
    def _setup_layout(self):
        """Set up the layout for the progress display."""
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="audio", size=15),  # Increased size for more tasks
            Layout(name="transcription", size=15),  # Increased size for more tasks
            Layout(name="summarization", size=15),  # Increased size for more tasks
            Layout(name="footer", size=3)
        )
        
        # Initialize all panels with placeholder content
        self.layout["header"].update(Panel(
            "[bold blue]Call Summarizer Pipeline[/bold blue]",
            border_style="blue"
        ))
        
        self.layout["audio"].update(Panel(
            "[dim]Audio Extraction - Waiting...[/dim]",
            title="Audio Extraction",
            border_style="green"
        ))
        
        self.layout["transcription"].update(Panel(
            "[dim]No audio files found[/dim]",
            title="Transcription", 
            border_style="yellow"
        ))
        
        self.layout["summarization"].update(Panel(
            "[dim]No transcript files found[/dim]",
            title="Summarization",
            border_style="magenta"
        ))
        
        self.layout["footer"].update(Panel(
            "[dim]Press Ctrl+C to cancel[/dim]",
            border_style="dim"
        ))
    
    def start(self):
        """Start the live progress display."""
        self.live = Live(self.layout, console=self.console, refresh_per_second=10, transient=True)
        self.live.start()
    
    def stop(self):
        """Stop the live progress display and clean up."""
        if self.live:
            self.live.stop()
            self.live = None
    
    def setup_audio_stage(self, total_files: int):
        """Initialize the audio extraction stage."""
        with self._lock:
            if total_files > 0:
                self.layout["audio"].update(Panel(
                    self.audio_progress,
                    title="Audio Extraction",
                    border_style="green"
                ))
                self.audio_progress.start()
            else:
                self.layout["audio"].update(Panel(
                    "[dim]No video files found[/dim]",
                    title="Audio Extraction",
                    border_style="green"
                ))
    
    def setup_transcription_stage(self, total_files: int):
        """Initialize the transcription stage."""
        with self._lock:
            if total_files > 0:
                self.layout["transcription"].update(Panel(
                    self.transcription_progress,
                    title="Transcription",
                    border_style="yellow"
                ))
                self.transcription_progress.start()
            else:
                self.layout["transcription"].update(Panel(
                    "[dim]No audio files found[/dim]",
                    title="Transcription",
                    border_style="yellow"
                ))
    
    def update_transcription_stage(self, total_files: int):
        """Update the transcription stage after audio extraction."""
        with self._lock:
            if total_files > 0:
                # Clear existing tasks
                self.transcription_tasks.clear()
                # Create a new progress instance for the updated stage
                self.transcription_progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(style="blue"),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    console=self.console,
                    expand=True,
                    refresh_per_second=10
                )
                self.transcription_progress.start()
                self.layout["transcription"].update(Panel(
                    self.transcription_progress,
                    title="Transcription",
                    border_style="yellow"
                ))
            else:
                self.transcription_tasks.clear()
                self.layout["transcription"].update(Panel(
                    "[dim]No audio files found[/dim]",
                    title="Transcription",
                    border_style="yellow"
                ))
    
    def setup_summarization_stage(self, total_files: int):
        """Initialize the summarization stage."""
        with self._lock:
            if total_files > 0:
                self.layout["summarization"].update(Panel(
                    self.summarization_progress,
                    title="Summarization",
                    border_style="magenta"
                ))
                self.summarization_progress.start()
            else:
                self.layout["summarization"].update(Panel(
                    "[dim]No transcript files found[/dim]",
                    title="Summarization",
                    border_style="magenta"
                ))
    
    def update_summarization_stage(self, total_files: int):
        """Update the summarization stage after transcription."""
        with self._lock:
            if total_files > 0:
                # Clear existing tasks
                self.summarization_tasks.clear()
                # Create a new progress instance for the updated stage
                self.summarization_progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(style="blue"),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    console=self.console,
                    expand=True,
                    refresh_per_second=10
                )
                self.summarization_progress.start()
                self.layout["summarization"].update(Panel(
                    self.summarization_progress,
                    title="Summarization",
                    border_style="magenta"
                ))
            else:
                self.summarization_tasks.clear()
                self.layout["summarization"].update(Panel(
                    "[dim]No transcript files found[/dim]",
                    title="Summarization",
                    border_style="magenta"
                ))
    
    def add_audio_task(self, filename: str, total_size: int = 0) -> TaskID:
        """Add a new audio extraction task."""
        with self._lock:
            task_id = self.audio_progress.add_task(
                f"Extracting: {filename}",
                total=total_size if total_size > 0 else 1
            )
            self.audio_tasks[filename] = task_id
            return task_id
    
    def add_transcription_task(self, filename: str, total_chunks: int = 1) -> TaskID:
        """Add a new transcription task."""
        with self._lock:
            task_id = self.transcription_progress.add_task(
                f"Transcribing: {filename}",
                total=total_chunks
            )
            self.transcription_tasks[filename] = task_id
            return task_id
    
    def add_summarization_task(self, filename: str) -> TaskID:
        """Add a new summarization task."""
        with self._lock:
            task_id = self.summarization_progress.add_task(
                f"Summarizing: {filename}",
                total=1
            )
            self.summarization_tasks[filename] = task_id
            return task_id
    
    def update_audio_task(self, filename: str, completed: int, description: str | None = None):
        """Update an audio extraction task."""
        if filename in self.audio_tasks:
            task_id = self.audio_tasks[filename]
            with self._lock:
                self.audio_progress.update(
                    task_id,
                    completed=completed,
                    description=description or f"Extracting: {filename}"
                )
    
    def update_transcription_task(self, filename: str, completed: int, description: str | None = None, total: int | None = None):
        """Update a transcription task."""
        if filename in self.transcription_tasks:
            task_id = self.transcription_tasks[filename]
            with self._lock:
                update_kwargs = {
                    "completed": completed,
                    "description": description or f"Transcribing: {filename}"
                }
                if total is not None:
                    update_kwargs["total"] = total
                self.transcription_progress.update(task_id, **update_kwargs)
    
    def update_transcription_chunk(self, filename: str, current_chunk: int, total_chunks: int):
        """Update transcription progress for a specific chunk."""
        if filename in self.transcription_tasks:
            task_id = self.transcription_tasks[filename]
            with self._lock:
                self.transcription_progress.update(
                    task_id,
                    completed=current_chunk,
                    total=total_chunks,
                    description=f"Transcribing: {filename} (chunk {current_chunk}/{total_chunks})"
                )
    
    def update_summarization_task(self, filename: str, completed: int, description: str | None = None):
        """Update a summarization task."""
        if filename in self.summarization_tasks:
            task_id = self.summarization_tasks[filename]
            with self._lock:
                self.summarization_progress.update(
                    task_id,
                    completed=completed,
                    description=description or f"Summarizing: {filename}"
                )
    
    def complete_audio_task(self, filename: str, success: bool = True):
        """Mark an audio task as completed."""
        if filename in self.audio_tasks:
            task_id = self.audio_tasks[filename]
            with self._lock:
                status = "✅" if success else "❌"
                # Get the total for this task and set completed to match
                task = self.audio_progress.tasks[task_id]
                self.audio_progress.update(
                    task_id,
                    completed=task.total,
                    description=f"{status} {filename}"
                )
    
    def complete_transcription_task(self, filename: str, success: bool = True):
        """Mark a transcription task as completed."""
        if filename in self.transcription_tasks:
            task_id = self.transcription_tasks[filename]
            with self._lock:
                status = "✅" if success else "❌"
                # Get the total for this task and set completed to match
                task = self.transcription_progress.tasks[task_id]
                self.transcription_progress.update(
                    task_id,
                    completed=task.total,
                    description=f"{status} {filename}"
                )
    
    def complete_summarization_task(self, filename: str, success: bool = True):
        """Mark a summarization task as completed."""
        if filename in self.summarization_tasks:
            task_id = self.summarization_tasks[filename]
            with self._lock:
                status = "✅" if success else "❌"
                # Get the total for this task and set completed to match
                task = self.summarization_progress.tasks[task_id]
                self.summarization_progress.update(
                    task_id,
                    completed=task.total,
                    description=f"{status} {filename}"
                )
    
    def show_final_summary(self, results: Dict[str, Any]):
        """Show a final summary of the pipeline results."""
        table = Table(title="Pipeline Summary")
        table.add_column("Stage", style="cyan")
        table.add_column("Successful", style="green")
        table.add_column("Failed", style="red")
        table.add_column("Skipped", style="yellow")
        table.add_column("Total", style="blue")
        table.add_column("Success Rate", style="magenta")
        
        for stage, summary in results.items():
            table.add_row(
                stage.title(),
                str(summary.successful),
                str(summary.failed),
                str(summary.skipped),
                str(summary.total),
                f"{summary.success_rate:.1f}%"
            )
        
        self.console.print("\n")
        self.console.print(table)
