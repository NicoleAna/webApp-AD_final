function toggleCheckboxes() {
    var checkboxs = document.querySelectorAll('input[type="checkbox"][name="algo"]');
    var selectAllCheckbox = document.getElementById('select-all-checkbox');

    checkboxs.forEach(function(checkbox) {
        checkbox.checked = selectAllCheckbox.checked;
    }); 
}

const dataItems = document.querySelectorAll('.data-item');

dataItems.forEach(item => {
    item.addEventListener('click', () => {
        const isItemOpen = item.classList.contains('active');
        dataItems.forEach(item => item.classList.remove('active'));
        if(!isItemOpen) {
            item.classList.toggle('active');
        }
    });
});

const modelCards = document.querySelectorAll('.model-card');

modelCards.forEach(card => {
  card.addEventListener('mouseenter', () => {
    card.classList.add('flip');
  });
  card.addEventListener('mouseleave', () => {
    card.classList.remove('flip');
  });
});