// Get references to the elements
const stars = document.getElementById('stars');
const moon = document.getElementById('moon');
const mountains_behind = document.getElementById('mountains_behind');
const text = document.getElementById('text');
const btn = document.getElementById('btn');
const mountains_front = document.getElementById('mountains_front');
const header = document.querySelector('header');

// Add scroll event listener
window.addEventListener('scroll', function() {
  // Get the current scroll position
  let value = window.scrollY;
  
  // Update element styles based on scroll position
  stars.style.left = value * 0.25 + 'px';
  moon.style.top = value * 1.05 + 'px';
  mountains_behind.style.top = value * 0.5 + 'px';
  mountains_front.style.top = value * 0 + 'px';
  text.style.marginRight = value * 4 + 'px';
  text.style.marginTop = value * 1.5 + 'px';
  btn.style.marginTop = value * 1.5 + 'px';
  header.style.top = value * 0.5 + 'px';
});
class IntersectionObserverList {
  mapping;
  observer;
  constructor() {
    this.mapping = new Map();
    this.observer = new IntersectionObserver(
      (entries) => {
        for (var entry of entries) {
          var callback = this.mapping.get(entry.target);

          callback && callback(entry.isIntersecting);
        }
      },
      {
        rootMargin: "300px 0px 300px 0px"
      }
    );
  }
  add(element, callback) {
    this.mapping.set(element, callback);
    this.observer.observe(element);
  }
  ngOnDestroy() {
    this.mapping.clear();
    this.observer.disconnect();
  }
  remove(element) {
    this.mapping.delete(element);
    this.observer.unobserve(element);
  }
}
const observer = new IntersectionObserverList();

$(window).mousemove(function (e) {
  $(".ring").css(
    "transform",
    `translateX(calc(${e.clientX}px - 1.25rem)) translateY(calc(${e.clientY}px - 1.25rem))`
  );
});

$('[data-animate="true"]').each(function (i) {
  console.log("$(this)", $(this));
  var element = $(this)[0];
  observer.add(element, (isIntersecting) => {
    if (isIntersecting) {
      $(this).addClass("animate-slide-down");
    } else {
      $(this).removeClass("animate-slide-down");
    }
  });
});