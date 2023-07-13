/*
    Front-end code for handling help info interactivity.
*/

/* Help controls */
const helpElem = document.getElementById('help');
const helpContentElem = document.getElementById('help-content');

helpElem.addEventListener('click', ev => {
    helpContentElem.style.display = helpContentElem.style.display == 'none' ? 'block' : 'none';
});