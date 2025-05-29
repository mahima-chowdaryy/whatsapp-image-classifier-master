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
    const form = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const resultContainer = document.getElementById('resultContainer');
    const resultImage = document.getElementById('resultImage');
    const categoryResult = document.getElementById('categoryResult');
    const documentProb = document.getElementById('documentProb');
    const nonGenericProb = document.getElementById('nonGenericProb');
    const genericProb = document.getElementById('genericProb');
    const documentValue = document.getElementById('documentValue');
    const nonGenericValue = document.getElementById('nonGenericValue');
    const genericValue = document.getElementById('genericValue');
    
    // Theme toggle
    const themeToggle = document.getElementById('theme-toggle');
    const themeIcon = themeToggle.querySelector('i');
    
    // Load saved theme
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        document.body.setAttribute('data-theme', savedTheme);
        themeIcon.className = savedTheme === 'light' ? 'fas fa-sun' : 'fas fa-moon';
    }

    themeToggle.addEventListener('click', () => {
        const currentTheme = document.body.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        document.body.setAttribute('data-theme', newTheme);
        themeIcon.className = newTheme === 'light' ? 'fas fa-sun' : 'fas fa-moon';
        localStorage.setItem('theme', newTheme);
    });

    // File input change handler
    fileInput.addEventListener('change', () => {
        const file = fileInput.files[0];
        if (file) {
            // Create a preview URL for the selected image
            const previewUrl = URL.createObjectURL(file);
            resultImage.src = previewUrl;
        }
    });

    // Form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const file = fileInput.files[0];
        if (!file) {
            showError('Please select an image first');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/classify', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                showResult(data);
            } else {
                showError(data.error || 'An error occurred during classification');
            }
        } catch (error) {
            console.error('Error:', error);
            showError('An error occurred while processing your request');
        }
    });

    function showResult(data) {
        // Update category
        categoryResult.textContent = data.result;
        categoryResult.className = `category-${data.result.toLowerCase()}`;

        // Update probability bars
        const probs = data.probabilities;
        documentProb.style.width = `${probs.document * 100}%`;
        nonGenericProb.style.width = `${probs.non_generic * 100}%`;
        genericProb.style.width = `${probs.generic * 100}%`;

        // Update probability values
        documentValue.textContent = `${(probs.document * 100).toFixed(1)}%`;
        nonGenericValue.textContent = `${(probs.non_generic * 100).toFixed(1)}%`;
        genericValue.textContent = `${(probs.generic * 100).toFixed(1)}%`;

        // Show result container
        resultContainer.style.display = 'block';
        resultContainer.scrollIntoView({ behavior: 'smooth' });
    }

    function showError(message) {
        const flashMessages = document.querySelector('.flash-messages');
        if (!flashMessages) {
            const container = document.createElement('div');
            container.className = 'flash-messages';
            document.querySelector('main').appendChild(container);
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = 'flash-message';
        messageDiv.innerHTML = `
            <i class="fas fa-exclamation-circle"></i>
            <span>${message}</span>
        `;

        document.querySelector('.flash-messages').appendChild(messageDiv);

        // Remove message after 5 seconds
        setTimeout(() => {
            messageDiv.remove();
        }, 5000);
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

// Image Preview Functionality
const fileInput = document.getElementById('file');
const previewContainer = document.getElementById('previewContainer');
const imagePreviews = document.getElementById('imagePreviews');
const uploadForm = document.getElementById('uploadForm');

if (fileInput) {
    fileInput.addEventListener('change', function(e) {
        const files = e.target.files;
        if (files.length > 0) {
            previewContainer.style.display = 'block';
            imagePreviews.innerHTML = '';
            
            Array.from(files).forEach((file, index) => {
                if (file.type.startsWith('image/')) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        const previewItem = document.createElement('div');
                        previewItem.className = 'preview-item';
                        previewItem.innerHTML = `
                            <img src="${e.target.result}" alt="Preview">
                            <button class="remove-btn" onclick="removeImage(${index})">
                                <i class="fas fa-times"></i>
                            </button>
                        `;
                        imagePreviews.appendChild(previewItem);
                    };
                    reader.readAsDataURL(file);
                }
            });
        } else {
            previewContainer.style.display = 'none';
        }
    });
}

function removeImage(index) {
    const dt = new DataTransfer();
    const files = fileInput.files;
    
    for (let i = 0; i < files.length; i++) {
        if (i !== index) {
            dt.items.add(files[i]);
        }
    }
    
    fileInput.files = dt.files;
    fileInput.dispatchEvent(new Event('change'));
}

// Form Submission with Preview
if (uploadForm) {
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData();
        const files = fileInput.files;
        
        if (files.length === 0) {
            showError('Please select at least one image');
            return;
        }
        
        // Validate file types
        const invalidFiles = Array.from(files).filter(file => {
            const extension = file.name.split('.').pop().toLowerCase();
            return !['jpg', 'jpeg', 'png'].includes(extension);
        });
        
        if (invalidFiles.length > 0) {
            showError('Invalid file type. Please upload only JPG, JPEG, or PNG images.');
            return;
        }
        
        Array.from(files).forEach(file => {
            formData.append('files[]', file);
        });
        
        // Show loading state
        const submitBtn = uploadForm.querySelector('.submit-btn');
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        submitBtn.disabled = true;
        
        // Hide any previous error
        const errorMessage = document.getElementById('errorMessage');
        errorMessage.style.display = 'none';
        
        fetch('/batch-classify', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'An error occurred while processing the images');
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                showError(data.error);
            } else {
                displayResults(data.results);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showError(error.message || 'An error occurred while processing the images');
        })
        .finally(() => {
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
        });
    });
}

function displayResults(results) {
    const resultSection = document.getElementById('resultSection');
    const resultsContainer = document.getElementById('resultsContainer');
    
    if (!resultSection || !resultsContainer) {
        console.error('Result elements not found');
        return;
    }
    
    resultsContainer.innerHTML = '';
    
    if (!Array.isArray(results)) {
        console.error('Invalid results format:', results);
        showError('Invalid response format from server');
        return;
    }
    
    results.forEach(result => {
        if (!result || typeof result !== 'object') {
            console.error('Invalid result item:', result);
            return;
        }
        
        const resultItem = document.createElement('div');
        resultItem.className = 'result-item';
        
        // Safely access properties with fallbacks
        const category = result.result || 'Unknown';
        const probabilities = result.probabilities || {};
        const imageUrl = result.image_url || '';
        
        resultItem.innerHTML = `
            <img src="${imageUrl}" alt="Classified Image" onerror="this.src='static/images/placeholder.png'">
            <div class="result-info">
                <h4>Category: ${category}</h4>
                <div class="probability-bars">
                    <div class="probability-item">
                        <span class="probability-label">Document</span>
                        <div class="probability-bar">
                            <div class="probability-fill" style="width: ${(probabilities.document || 0) * 100}%"></div>
                        </div>
                        <span class="probability-value">${((probabilities.document || 0) * 100).toFixed(1)}%</span>
                    </div>
                    <div class="probability-item">
                        <span class="probability-label">Non-Generic</span>
                        <div class="probability-bar">
                            <div class="probability-fill" style="width: ${(probabilities.non_generic || 0) * 100}%"></div>
                        </div>
                        <span class="probability-value">${((probabilities.non_generic || 0) * 100).toFixed(1)}%</span>
                    </div>
                    <div class="probability-item">
                        <span class="probability-label">Generic</span>
                        <div class="probability-bar">
                            <div class="probability-fill" style="width: ${(probabilities.generic || 0) * 100}%"></div>
                        </div>
                        <span class="probability-value">${((probabilities.generic || 0) * 100).toFixed(1)}%</span>
                    </div>
                </div>
            </div>
        `;
        resultsContainer.appendChild(resultItem);
    });
    
    resultSection.style.display = 'block';
    resultSection.scrollIntoView({ behavior: 'smooth' });
}

// Theme Persistence
const themeToggle = document.getElementById('themeToggle');
if (themeToggle) {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.body.setAttribute('data-theme', savedTheme);
    themeToggle.checked = savedTheme === 'dark';
    
    themeToggle.addEventListener('change', function() {
        const theme = this.checked ? 'dark' : 'light';
        document.body.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
    });
}

// Error Handling
function showError(message) {
    const errorMessage = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');
    
    if (!errorMessage || !errorText) {
        console.error('Error message elements not found');
        return;
    }
    
    errorText.textContent = message;
    errorMessage.style.display = 'flex';
    
    // Scroll to error message
    errorMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });
    
    // Hide error after 5 seconds
    setTimeout(() => {
        errorMessage.style.display = 'none';
    }, 5000);
}

// Sample Images Functionality
document.addEventListener('DOMContentLoaded', () => {
    const sampleItems = document.querySelectorAll('.sample-item');
    
    sampleItems.forEach(item => {
        item.addEventListener('click', () => {
            const img = item.querySelector('img');
            const fileInput = document.getElementById('fileInput');
            
            // Create a fetch request to get the image as a blob
            fetch(img.src)
                .then(response => response.blob())
                .then(blob => {
                    // Create a File object from the blob
                    const file = new File([blob], 'sample.jpg', { type: 'image/jpeg' });
                    
                    // Create a new FileList-like object
                    const dataTransfer = new DataTransfer();
                    dataTransfer.items.add(file);
                    fileInput.files = dataTransfer.files;
                    
                    // Trigger the file input change event
                    const event = new Event('change', { bubbles: true });
                    fileInput.dispatchEvent(event);
                    
                    // Submit the form
                    document.getElementById('uploadForm').dispatchEvent(new Event('submit'));
                })
                .catch(error => {
                    console.error('Error loading sample image:', error);
                    showError('Error loading sample image');
                });
        });
    });
}); 