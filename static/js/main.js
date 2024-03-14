function toggleCheckboxes() {
    var checkboxs = document.querySelectorAll('input[type="checkbox"][name="algo"]');
    var selectAllCheckbox = document.getElementById('select-all-checkbox');

    checkboxs.forEach(function(checkbox) {
        checkbox.checked = selectAllCheckbox.checked;
    }); 
}