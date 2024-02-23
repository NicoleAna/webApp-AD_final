function toggleCheckboxes() {
    var checkboxs = document.querySelectorAll('input[type="checkbox"][name="algo"]');
    var selectAllCheckbox = document.getElementById('select-all-checkbox');

    checkboxs.forEach(function(checkbox) {
        checkbox.checked = selectAllCheckbox.checked;
    }); 
}

function showForm(option) {
    var opt1 = document.getElementById('uni-form')
    var opt2 = document.getElementById('multi-form')
    var opt = document.getElementById('opt-form')

    if (option == 'uni') {
        opt1.style.display = "block"
        opt2.style.display = "none"
        opt.style.display = "none"
    } else if (option == 'multi') {
        opt1.style.display = "none"
        opt2.style.display = "block"
        opt.style.display = "none"
    }
}