/* Utils for starting and stopping a progress bar */


const pbElem = getElem("progressBar");

let pbInterval = null;
let pbWidth = 1;

if (pbElem != undefined) {
    function setPBWidth() {
        pbElem.style.width = pbWidth + "%";
    }

    function progressBarStep() {
        pbWidth++;
        if (pbWidth > 100) {
            pbWidth = 1;
        }
        setPBWidth();
    }

    function startProgressBar() {
        console.log('started progress bar')
        pbElem.style.display = 'block';
        pbInterval = setInterval(progressBarStep, 40);
    }

    function stopProgressBar() {
        console.log('stopped progress bar')
        if (pbInterval != null) {
            clearInterval(pbInterval);
        }
        pbWidth = 1;
        setPBWidth();
        pbElem.style.display = 'none';
    }
}

