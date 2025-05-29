// Theme Toggle Functionality
document.addEventListener('DOMContentLoaded', () => {
    const themeToggle = document.getElementById('theme-toggle');
    const themeIcon = themeToggle.querySelector('i');
    
    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        document.documentElement.setAttribute('data-theme', savedTheme);
        updateThemeIcon(savedTheme);
    }
    
    themeToggle.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        updateThemeIcon(newTheme);
    });
    
    function updateThemeIcon(theme) {
        themeIcon.className = theme === 'light' ? 'fas fa-sun' : 'fas fa-moon';
    }
});

// File Upload and Classification Handling
document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('file');
    const resultSection = document.getElementById('resultSection');
    const errorMessage = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');
    
    // Update file input label when file is selected
    fileInput.addEventListener('change', (e) => {
        const fileName = e.target.files[0]?.name;
        const fileLabel = document.querySelector('.file-label');
        if (fileName) {
            fileLabel.querySelector('.upload-text').textContent = fileName;
        } else {
            fileLabel.querySelector('.upload-text').textContent = 'Choose a file or drag it here';
        }
    });
    
    // Handle form submission
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const file = fileInput.files[0];
        if (!file) {
            showError('Please select a file first');
            return;
        }
        
        // Show loading state
        const submitBtn = uploadForm.querySelector('.submit-btn');
        const originalBtnText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        submitBtn.disabled = true;
        
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/classify', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok) {
                displayResults(data);
            } else {
                showError(data.error || 'An error occurred during classification');
            }
        } catch (error) {
            showError('Failed to process the image. Please try again.');
            console.error('Error:', error);
        } finally {
            // Reset button state
            submitBtn.innerHTML = originalBtnText;
            submitBtn.disabled = false;
        }
    });
    
    function displayResults(data) {
        // Hide any previous error
        errorMessage.style.display = 'none';
        
        // Show result section
        resultSection.style.display = 'block';
        
        // Update result category
        const resultCategory = document.getElementById('resultCategory');
        resultCategory.textContent = data.result;
        resultCategory.className = 'result-category ' + data.result.toLowerCase();
        
        // Update probability bars
        updateProbabilityBar('documentProb', 'documentValue', data.probabilities.document);
        updateProbabilityBar('nonGenericProb', 'nonGenericValue', data.probabilities.non_generic);
        updateProbabilityBar('genericProb', 'genericValue', data.probabilities.generic);
        
        // Scroll to results
        resultSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    function updateProbabilityBar(barId, valueId, probability) {
        const bar = document.getElementById(barId);
        const value = document.getElementById(valueId);
        const percentage = (probability * 100).toFixed(1);
        
        bar.style.width = percentage + '%';
        value.textContent = percentage + '%';
    }
    
    function showError(message) {
        errorText.textContent = message;
        errorMessage.style.display = 'flex';
        resultSection.style.display = 'none';
    }
});

// Recent Classifications Animation
document.addEventListener('DOMContentLoaded', () => {
    const recentItems = document.querySelectorAll('.recent-item');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, {
        threshold: 0.1
    });
    
    recentItems.forEach(item => {
        item.style.opacity = '0';
        item.style.transform = 'translateY(20px)';
        observer.observe(item);
    });
}); 