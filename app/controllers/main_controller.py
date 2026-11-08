"""
main_controller.py — Read-only Flask routes.
All data is read from SQLite; no ML training happens here.
"""

from flask import Blueprint, render_template, redirect, url_for, flash
from app.models.db_manager import get_all_runs, get_run, get_latest_run_per_dataset
from app.models.dataset_loader import DATASETS as DS_CFG

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """Dashboard: list all datasets with their latest experiment summary."""
    latest_runs = {r['dataset_id']: r for r in get_latest_run_per_dataset()}
    datasets = list(DS_CFG.values())
    return render_template('index.html', datasets=datasets, latest_runs=latest_runs)


@main_bp.route('/dataset/<dataset_id>')
def dataset_view(dataset_id):
    """Full results page for the latest run of a given dataset."""
    runs = [r for r in get_all_runs() if r['dataset_id'] == dataset_id]
    if not runs:
        flash(f'No training results found for "{dataset_id}". '
              'Run train_and_save.py first.', 'warning')
        return redirect(url_for('main.index'))
    # Latest run is first (ORDER BY id DESC in get_all_runs)
    run = get_run(runs[0]['id'])
    return render_template('dataset_view.html', run=run, all_runs=runs)


@main_bp.route('/run/<int:run_id>')
def run_detail(run_id):
    """Full results page for any specific run by ID."""
    run = get_run(run_id)
    if not run:
        flash(f'Run #{run_id} not found.', 'warning')
        return redirect(url_for('main.history'))
    return render_template('dataset_view.html', run=run, all_runs=[])


@main_bp.route('/history')
def history():
    """Paginated log of all saved experiment runs."""
    runs = get_all_runs(limit=50)
    return render_template('history_view.html', runs=runs)
