self.addEventListener('install', function(event) {
  console.log('Service worker installed.');
});

self.addEventListener('activate', function(event) {
  console.log('Service worker activated.');
});

self.addEventListener('fetch', function(event) {
  console.log('Fetch event for ', event.request.url);
});