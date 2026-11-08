// Frontend helper interactions
document.addEventListener('DOMContentLoaded', function () {
    console.log('ML Pre-Learning MVC Platform loaded.');
    
    // Auto-scroll to config form when clicking quick action
    const inspectLinks = document.querySelectorAll('a[href^="#"]');
    inspectLinks.forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                e.preventDefault();
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });

    // Form submission loading state
    const trainForms = document.querySelectorAll('form');
    trainForms.forEach(form => {
        form.addEventListener('submit', function () {
            const submitBtn = this.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = true;
                const originalHtml = submitBtn.innerHTML;
                submitBtn.innerHTML = `
                    <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                    Running Pipeline & Training...
                `;
            }
        });
    });
});
