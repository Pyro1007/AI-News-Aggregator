document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const newsForm = document.getElementById('newsForm');
    const newsContainer = document.getElementById('newsContainer');
    const loadingIndicator = document.getElementById('loading');
    const errorAlert = document.getElementById('errorAlert');
    const noNewsMessage = document.getElementById('noNews');

    // Set default date to today
    document.getElementById('date').valueAsDate = new Date();

    // Form submission handler
    newsForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Clear previous results
        newsContainer.innerHTML = '';
        errorAlert.classList.add('d-none');
        noNewsMessage.classList.add('d-none');
        setLoadingState(true);

        try {
            const requestData = {
                category: document.getElementById('category').value,
                date: document.getElementById('date').value,
                translation: document.getElementById('translationToggle').checked,
                translation_language: document.getElementById('language').value
            };

            const response = await fetch('/get_news', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) throw new Error('Failed to fetch news');

            const data = await response.json();
            
            if (!data.news || data.news.length === 0) {
                showNoNewsMessage();
                return;
            }

            // Display news items one by one
            for (const item of data.news) {
                await displayNewsItem(item);
                await new Promise(resolve => setTimeout(resolve, 200)); // Small delay between items
            }

        } catch (error) {
            showError(error.message || 'Error fetching news');
        } finally {
            setLoadingState(false);
        }
    });

    // Display a single news item
    async function displayNewsItem(item) {
        return new Promise(resolve => {
            const newsCard = document.createElement('div');
            newsCard.className = 'col mb-4';
            newsCard.innerHTML = `
                <div class="card h-100 ${getSentimentClass(item.sentiment)}">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-start mb-2">
                            ${item.sentiment ? `<span class="badge ${getSentimentBadgeClass(item.sentiment)}">${item.sentiment}</span>` : ''}
                            <small class="text-muted">${formatDate(item.published_date || item.published)}</small>
                        </div>
                        <h5 class="card-title">${item.title}</h5>
                        <h6 class="card-subtitle mb-2 text-muted">${item.source || 'Unknown source'}</h6>
                        <p class="card-text">${item.summary || 'No summary available'}</p>
                        ${item.translated_summary ? `
                        <div class="translated-text mt-3">
                            <small class="text-muted">Translated to ${getLanguageName(item.translation_language)}:</small>
                            <p class="mt-1">${item.translated_summary}</p>
                        </div>
                        ` : ''}
                    </div>
                    <div class="card-footer bg-transparent">
                        <a href="${item.url}" target="_blank" class="btn btn-sm btn-outline-primary">Read Full Article</a>
                    </div>
                </div>
            `;

            // Add card to container with fade-in animation
            newsCard.style.opacity = '0';
            newsContainer.appendChild(newsCard);
            
            // Animate the appearance
            let opacity = 0;
            const fadeIn = setInterval(() => {
                opacity += 0.05;
                newsCard.style.opacity = opacity;
                if (opacity >= 1) {
                    clearInterval(fadeIn);
                    resolve();
                }
            }, 20);
        });
    }

    // Helper functions
    function formatDate(dateString) {
        if (!dateString) return '';
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    function getLanguageName(code) {
        const languages = {
            'hi': 'Hindi', 'mr': 'Marathi', 
            'es': 'Spanish', 'fr': 'French'
        };
        return languages[code] || code;
    }

    function getSentimentClass(sentiment) {
        return {
            'positive': 'border-left-success',
            'negative': 'border-left-danger',
            'neutral': 'border-left-warning'
        }[sentiment] || '';
    }

    function getSentimentBadgeClass(sentiment) {
        return {
            'positive': 'bg-success',
            'negative': 'bg-danger',
            'neutral': 'bg-warning text-dark'
        }[sentiment] || '';
    }

    function setLoadingState(isLoading) {
        document.getElementById('fetchBtn').disabled = isLoading;
        loadingIndicator.classList.toggle('d-none', !isLoading);
    }

    function showError(message) {
        errorAlert.textContent = message;
        errorAlert.classList.remove('d-none');
    }

    function showNoNewsMessage() {
        noNewsMessage.classList.remove('d-none');
    }
});
