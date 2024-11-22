// script.js
document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const loading = document.getElementById('loading');
    const fileInput = document.getElementById('file');
    const uploadText = document.querySelector('.upload-text');
    
    // File input change handler
    fileInput.addEventListener('change', function(e) {
        const file = this.files[0];
        
        if (file) {
            // Update upload text with file name
            uploadText.textContent = file.name;
            // Add active class to upload area
            document.querySelector('.file-label').classList.add('active');
            
            // Validate file type (optional)
            const validTypes = ['video/mp4', 'video/webm', 'video/ogg'];
            if (!validTypes.includes(file.type)) {
                alert('Please select a valid video file (MP4, WebM, or OGG)');
                this.value = '';
                uploadText.textContent = 'Select Video File';
                document.querySelector('.file-label').classList.remove('active');
            }
        } else {
            // Reset upload text if no file selected
            uploadText.textContent = 'Select Video File';
            document.querySelector('.file-label').classList.remove('active');
        }
    });
    
    // Form submit handler
    uploadForm.addEventListener('submit', function(e) {
        const file = fileInput.files[0];
        
        if (file) {
            // Hide the form and show loading animation
            uploadForm.style.display = 'none';
            loading.style.display = 'block';
            
            // Add loading text animation
            const loadingText = document.querySelector('.loading-text');
            let dots = '';
            setInterval(() => {
                dots = dots.length < 3 ? dots + '.' : '';
                loadingText.textContent = `Processing video with AI${dots}`;
            }, 500);
        }
    });
    
    // Drag and drop functionality
    const uploadArea = document.querySelector('.upload-area');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight(e) {
        uploadArea.classList.add('drag-highlight');
    }
    
    function unhighlight(e) {
        uploadArea.classList.remove('drag-highlight');
    }
    
    uploadArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const file = dt.files[0];
        
        fileInput.files = dt.files;
        if (file) {
            uploadText.textContent = file.name;
            document.querySelector('.file-label').classList.add('active');
        }
    }
});