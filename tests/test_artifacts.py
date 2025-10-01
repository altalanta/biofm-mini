"""Test artifact creation and validation for biofm-mini demo.

Tests verify that the demo pipeline creates the expected artifacts
and that they contain valid content with proper structure.
"""

import json
import subprocess
from pathlib import Path


class TestDemoArtifacts:
    """Test class for demo artifact validation."""
    
    def test_demo_creates_expected_files(self, tmp_path):
        """Test that running the demo creates all expected artifacts."""
        # Change to repo directory and run demo
        original_cwd = Path.cwd()
        repo_root = Path(__file__).parent.parent
        
        try:
            # Change to repo directory
            import os
            os.chdir(repo_root)
            
            # Run the demo script
            result = subprocess.run(
                ["python", "-m", "scripts.demo_end_to_end"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Check that demo ran successfully
            assert result.returncode == 0, f"Demo failed with stderr: {result.stderr}"
            
            # Check that artifacts directory exists
            artifacts_dir = repo_root / "artifacts" / "demo"
            assert artifacts_dir.exists(), "Artifacts directory not created"
            
            # Check for expected files
            expected_files = [
                "metrics.json",
                "roc_curve.png", 
                "pr_curve.png",
                "calibration_curve.png",
                "confusion_matrix.png",
                "version.txt"
            ]
            
            for filename in expected_files:
                file_path = artifacts_dir / filename
                assert file_path.exists(), f"Expected file {filename} not found"
                assert file_path.stat().st_size > 0, f"File {filename} is empty"
        
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
    
    def test_metrics_json_structure(self, tmp_path):
        """Test that metrics.json has the expected structure and content."""
        # Run demo and check metrics.json
        original_cwd = Path.cwd()
        repo_root = Path(__file__).parent.parent
        
        try:
            import os
            os.chdir(repo_root)
            
            # Run the demo
            result = subprocess.run(
                ["python", "-m", "scripts.demo_end_to_end"],
                capture_output=True,
                text=True,
                timeout=60
            )
            assert result.returncode == 0, f"Demo failed: {result.stderr}"
            
            # Load and validate metrics.json
            metrics_file = repo_root / "artifacts" / "demo" / "metrics.json"
            assert metrics_file.exists(), "metrics.json not found"
            
            with open(metrics_file) as f:
                metrics = json.load(f)
            
            # Check for required keys
            required_keys = ["auroc", "auprc", "accuracy", "precision", "recall", "f1", "ece"]
            for key in required_keys:
                assert key in metrics, f"Required metric {key} not found in metrics.json"
                assert isinstance(metrics[key], (int, float)), f"Metric {key} is not numeric"
                assert 0 <= metrics[key] <= 1, f"Metric {key} value {metrics[key]} out of range [0, 1]"
            
            # Validate specific metric ranges
            assert 0.5 <= metrics["auroc"] <= 1.0, f"AUROC {metrics['auroc']} seems unreasonable"
            assert 0.0 <= metrics["ece"] <= 0.5, f"ECE {metrics['ece']} seems too high"
            
        finally:
            os.chdir(original_cwd)
    
    def test_version_txt_content(self, tmp_path):
        """Test that version.txt contains expected version information."""
        original_cwd = Path.cwd()
        repo_root = Path(__file__).parent.parent
        
        try:
            import os
            os.chdir(repo_root)
            
            # Run the demo
            result = subprocess.run(
                ["python", "-m", "scripts.demo_end_to_end"],
                capture_output=True,
                text=True,
                timeout=60
            )
            assert result.returncode == 0, f"Demo failed: {result.stderr}"
            
            # Check version.txt content
            version_file = repo_root / "artifacts" / "demo" / "version.txt"
            assert version_file.exists(), "version.txt not found"
            
            content = version_file.read_text()
            lines = content.strip().split('\n')
            
            # Parse key-value pairs
            version_info = {}
            for line in lines:
                if '=' in line:
                    key, value = line.split('=', 1)
                    version_info[key] = value
            
            # Check for required keys
            required_keys = ["git_sha", "package_version", "seed"]
            for key in required_keys:
                assert key in version_info, f"Required version info {key} not found"
            
            # Validate seed value
            assert version_info["seed"] == "1337", f"Unexpected seed value: {version_info['seed']}"
            
        finally:
            os.chdir(original_cwd)
    
    def test_plot_files_are_valid_images(self, tmp_path):
        """Test that generated plot files are valid PNG images."""
        original_cwd = Path.cwd()
        repo_root = Path(__file__).parent.parent
        
        try:
            import os
            os.chdir(repo_root)
            
            # Run the demo
            result = subprocess.run(
                ["python", "-m", "scripts.demo_end_to_end"],
                capture_output=True,
                text=True,
                timeout=60
            )
            assert result.returncode == 0, f"Demo failed: {result.stderr}"
            
            # Check that plot files are valid PNG images
            artifacts_dir = repo_root / "artifacts" / "demo"
            plot_files = [
                "roc_curve.png",
                "pr_curve.png", 
                "calibration_curve.png",
                "confusion_matrix.png"
            ]
            
            for plot_file in plot_files:
                file_path = artifacts_dir / plot_file
                assert file_path.exists(), f"Plot file {plot_file} not found"
                
                # Check file size is reasonable (not too small, not too large)
                file_size = file_path.stat().st_size
                assert 1000 < file_size < 1_000_000, (
                    f"Plot file {plot_file} has suspicious size: {file_size} bytes"
                )
                
                # Check PNG header
                with open(file_path, 'rb') as f:
                    header = f.read(8)
                    png_signature = b'\x89PNG\r\n\x1a\n'
                    assert header == png_signature, f"File {plot_file} is not a valid PNG"
        
        finally:
            os.chdir(original_cwd)
    
    def test_demo_is_deterministic(self, tmp_path):
        """Test that running the demo twice produces identical artifacts."""
        original_cwd = Path.cwd()
        repo_root = Path(__file__).parent.parent
        
        try:
            import os
            os.chdir(repo_root)
            
            # Clean any existing artifacts
            artifacts_dir = repo_root / "artifacts" / "demo"
            if artifacts_dir.exists():
                import shutil
                shutil.rmtree(artifacts_dir)
            
            # Run demo first time
            result1 = subprocess.run(
                ["python", "-m", "scripts.demo_end_to_end"],
                capture_output=True,
                text=True,
                timeout=60
            )
            assert result1.returncode == 0, f"First demo run failed: {result1.stderr}"
            
            # Read first metrics
            metrics_file = artifacts_dir / "metrics.json"
            with open(metrics_file) as f:
                metrics1 = json.load(f)
            
            # Save first version file content
            version_file = artifacts_dir / "version.txt"
            version_content1 = version_file.read_text()
            
            # Clean artifacts
            import shutil
            shutil.rmtree(artifacts_dir)
            
            # Run demo second time
            result2 = subprocess.run(
                ["python", "-m", "scripts.demo_end_to_end"],
                capture_output=True,
                text=True,
                timeout=60
            )
            assert result2.returncode == 0, f"Second demo run failed: {result2.stderr}"
            
            # Read second metrics
            with open(metrics_file) as f:
                metrics2 = json.load(f)
            
            # Read second version file content
            version_content2 = version_file.read_text()
            
            # Compare metrics (should be identical within tight tolerance)
            tolerance = 1e-6
            for key in metrics1.keys():
                assert abs(metrics1[key] - metrics2[key]) < tolerance, (
                    f"Metric {key} not deterministic: {metrics1[key]} vs {metrics2[key]}"
                )
            
            # Version content should be identical (except possibly git_sha if commits changed)
            lines1 = version_content1.strip().split('\n')
            lines2 = version_content2.strip().split('\n')
            
            for line1, line2 in zip(lines1, lines2):
                if line1.startswith('seed='):
                    assert line1 == line2, f"Seed line differs: {line1} vs {line2}"
                # Allow git_sha and package_version to potentially differ
        
        finally:
            os.chdir(original_cwd)


class TestDemoPerformance:
    """Test class for demo performance characteristics."""
    
    def test_demo_runtime_within_limits(self, tmp_path):
        """Test that demo completes within reasonable time limits."""
        import time
        original_cwd = Path.cwd()
        repo_root = Path(__file__).parent.parent
        
        try:
            import os
            os.chdir(repo_root)
            
            # Time the demo execution
            start_time = time.time()
            result = subprocess.run(
                ["python", "-m", "scripts.demo_end_to_end"],
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            end_time = time.time()
            
            assert result.returncode == 0, f"Demo failed: {result.stderr}"
            
            runtime = end_time - start_time
            # Demo should complete in under 30 seconds on most systems
            assert runtime < 30, f"Demo took too long: {runtime:.2f} seconds"
            
            # Demo should not be suspiciously fast (at least 0.5 seconds)
            assert runtime > 0.5, f"Demo completed suspiciously fast: {runtime:.2f} seconds"
        
        finally:
            os.chdir(original_cwd)
    
    def test_clean_artifacts_target(self, tmp_path):
        """Test that clean-artifacts make target works correctly."""
        original_cwd = Path.cwd()
        repo_root = Path(__file__).parent.parent
        
        try:
            import os
            os.chdir(repo_root)
            
            # Run demo to create artifacts
            result = subprocess.run(
                ["python", "-m", "scripts.demo_end_to_end"],
                capture_output=True,
                text=True,
                timeout=60
            )
            assert result.returncode == 0, f"Demo failed: {result.stderr}"
            
            # Verify artifacts exist
            artifacts_dir = repo_root / "artifacts" / "demo"
            assert artifacts_dir.exists(), "Artifacts directory should exist"
            
            # Run clean-artifacts
            result = subprocess.run(
                ["make", "clean-artifacts"],
                capture_output=True,
                text=True,
                timeout=10
            )
            assert result.returncode == 0, f"clean-artifacts failed: {result.stderr}"
            
            # Verify artifacts are cleaned
            assert not artifacts_dir.exists(), "Artifacts directory should be removed"
        
        finally:
            os.chdir(original_cwd)