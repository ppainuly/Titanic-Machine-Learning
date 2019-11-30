# Titanic-Machine-Learning
Predicting Survivors from Titanic Dataset

<!DOCTYPE html>
<html>
<head><meta charset="utf-8" />

<title>Titanic_Survivor_Prediction</title>

<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.3/jquery.min.js"></script>



<style type="text/css">
    /*!
*
* Twitter Bootstrap
*
*/
/*!
 * Bootstrap v3.3.7 (http://getbootstrap.com)
 * Copyright 2011-2016 Twitter, Inc.
 * Licensed under MIT (https://github.com/twbs/bootstrap/blob/master/LICENSE)
 */
/*! normalize.css v3.0.3 | MIT License | github.com/necolas/normalize.css */
html {
  font-family: sans-serif;
  -ms-text-size-adjust: 100%;
  -webkit-text-size-adjust: 100%;
}
body {
  margin: 0;
}
article,
aside,
details,
figcaption,
figure,
footer,
header,
hgroup,
main,
menu,
nav,
section,
summary {
  display: block;
}
audio,
canvas,
progress,
video {
  display: inline-block;
  vertical-align: baseline;
}
audio:not([controls]) {
  display: none;
  height: 0;
}
[hidden],
template {
  display: none;
}
a {
  background-color: transparent;
}
a:active,
a:hover {
  outline: 0;
}
abbr[title] {
  border-bottom: 1px dotted;
}
b,
strong {
  font-weight: bold;
}
dfn {
  font-style: italic;
}
h1 {
  font-size: 2em;
  margin: 0.67em 0;
}
mark {
  background: #ff0;
  color: #000;
}
small {
  font-size: 80%;
}
sub,
sup {
  font-size: 75%;
  line-height: 0;
  position: relative;
  vertical-align: baseline;
}
sup {
  top: -0.5em;
}
sub {
  bottom: -0.25em;
}
img {
  border: 0;
}
svg:not(:root) {
  overflow: hidden;
}
figure {
  margin: 1em 40px;
}
hr {
  box-sizing: content-box;
  height: 0;
}
pre {
  overflow: auto;
}
code,
kbd,
pre,
samp {
  font-family: monospace, monospace;
  font-size: 1em;
}
button,
input,
optgroup,
select,
textarea {
  color: inherit;
  font: inherit;
  margin: 0;
}
button {
  overflow: visible;
}
button,
select {
  text-transform: none;
}
button,
html input[type="button"],
input[type="reset"],
input[type="submit"] {
  -webkit-appearance: button;
  cursor: pointer;
}
button[disabled],
html input[disabled] {
  cursor: default;
}
button::-moz-focus-inner,
input::-moz-focus-inner {
  border: 0;
  padding: 0;
}
input {
  line-height: normal;
}
input[type="checkbox"],
input[type="radio"] {
  box-sizing: border-box;
  padding: 0;
}
input[type="number"]::-webkit-inner-spin-button,
input[type="number"]::-webkit-outer-spin-button {
  height: auto;
}
input[type="search"] {
  -webkit-appearance: textfield;
  box-sizing: content-box;
}
input[type="search"]::-webkit-search-cancel-button,
input[type="search"]::-webkit-search-decoration {
  -webkit-appearance: none;
}
fieldset {
  border: 1px solid #c0c0c0;
  margin: 0 2px;
  padding: 0.35em 0.625em 0.75em;
}
legend {
  border: 0;
  padding: 0;
}
textarea {
  overflow: auto;
}
optgroup {
  font-weight: bold;
}
table {
  border-collapse: collapse;
  border-spacing: 0;
}
td,
th {
  padding: 0;
}
/*! Source: https://github.com/h5bp/html5-boilerplate/blob/master/src/css/main.css */
@media print {
  *,
  *:before,
  *:after {
    background: transparent !important;
    box-shadow: none !important;
    text-shadow: none !important;
  }
  a,
  a:visited {
    text-decoration: underline;
  }
  a[href]:after {
    content: " (" attr(href) ")";
  }
  abbr[title]:after {
    content: " (" attr(title) ")";
  }
  a[href^="#"]:after,
  a[href^="javascript:"]:after {
    content: "";
  }
  pre,
  blockquote {
    border: 1px solid #999;
    page-break-inside: avoid;
  }
  thead {
    display: table-header-group;
  }
  tr,
  img {
    page-break-inside: avoid;
  }
  img {
    max-width: 100% !important;
  }
  p,
  h2,
  h3 {
    orphans: 3;
    widows: 3;
  }
  h2,
  h3 {
    page-break-after: avoid;
  }
  .navbar {
    display: none;
  }
  .btn > .caret,
  .dropup > .btn > .caret {
    border-top-color: #000 !important;
  }
  .label {
    border: 1px solid #000;
  }
  .table {
    border-collapse: collapse !important;
  }
  .table td,
  .table th {
    background-color: #fff !important;
  }
  .table-bordered th,
  .table-bordered td {
    border: 1px solid #ddd !important;
  }
}
@font-face {
  font-family: 'Glyphicons Halflings';
  src: url('../components/bootstrap/fonts/glyphicons-halflings-regular.eot');
  src: url('../components/bootstrap/fonts/glyphicons-halflings-regular.eot?#iefix') format('embedded-opentype'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.woff2') format('woff2'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.woff') format('woff'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.ttf') format('truetype'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.svg#glyphicons_halflingsregular') format('svg');
}
.glyphicon {
  position: relative;
  top: 1px;
  display: inline-block;
  font-family: 'Glyphicons Halflings';
  font-style: normal;
  font-weight: normal;
  line-height: 1;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
.glyphicon-asterisk:before {
  content: "\002a";
}
.glyphicon-plus:before {
  content: "\002b";
}
.glyphicon-euro:before,
.glyphicon-eur:before {
  content: "\20ac";
}
.glyphicon-minus:before {
  content: "\2212";
}
.glyphicon-cloud:before {
  content: "\2601";
}
.glyphicon-envelope:before {
  content: "\2709";
}
.glyphicon-pencil:before {
  content: "\270f";
}
.glyphicon-glass:before {
  content: "\e001";
}
.glyphicon-music:before {
  content: "\e002";
}
.glyphicon-search:before {
  content: "\e003";
}
.glyphicon-heart:before {
  content: "\e005";
}
.glyphicon-star:before {
  content: "\e006";
}
.glyphicon-star-empty:before {
  content: "\e007";
}
.glyphicon-user:before {
  content: "\e008";
}
.glyphicon-film:before {
  content: "\e009";
}
.glyphicon-th-large:before {
  content: "\e010";
}
.glyphicon-th:before {
  content: "\e011";
}
.glyphicon-th-list:before {
  content: "\e012";
}
.glyphicon-ok:before {
  content: "\e013";
}
.glyphicon-remove:before {
  content: "\e014";
}
.glyphicon-zoom-in:before {
  content: "\e015";
}
.glyphicon-zoom-out:before {
  content: "\e016";
}
.glyphicon-off:before {
  content: "\e017";
}
.glyphicon-signal:before {
  content: "\e018";
}
.glyphicon-cog:before {
  content: "\e019";
}
.glyphicon-trash:before {
  content: "\e020";
}
.glyphicon-home:before {
  content: "\e021";
}
.glyphicon-file:before {
  content: "\e022";
}
.glyphicon-time:before {
  content: "\e023";
}
.glyphicon-road:before {
  content: "\e024";
}
.glyphicon-download-alt:before {
  content: "\e025";
}
.glyphicon-download:before {
  content: "\e026";
}
.glyphicon-upload:before {
  content: "\e027";
}
.glyphicon-inbox:before {
  content: "\e028";
}
.glyphicon-play-circle:before {
  content: "\e029";
}
.glyphicon-repeat:before {
  content: "\e030";
}
.glyphicon-refresh:before {
  content: "\e031";
}
.glyphicon-list-alt:before {
  content: "\e032";
}
.glyphicon-lock:before {
  content: "\e033";
}
.glyphicon-flag:before {
  content: "\e034";
}
.glyphicon-headphones:before {
  content: "\e035";
}
.glyphicon-volume-off:before {
  content: "\e036";
}
.glyphicon-volume-down:before {
  content: "\e037";
}
.glyphicon-volume-up:before {
  content: "\e038";
}
.glyphicon-qrcode:before {
  content: "\e039";
}
.glyphicon-barcode:before {
  content: "\e040";
}
.glyphicon-tag:before {
  content: "\e041";
}
.glyphicon-tags:before {
  content: "\e042";
}
.glyphicon-book:before {
  content: "\e043";
}
.glyphicon-bookmark:before {
  content: "\e044";
}
.glyphicon-print:before {
  content: "\e045";
}
.glyphicon-camera:before {
  content: "\e046";
}
.glyphicon-font:before {
  content: "\e047";
}
.glyphicon-bold:before {
  content: "\e048";
}
.glyphicon-italic:before {
  content: "\e049";
}
.glyphicon-text-height:before {
  content: "\e050";
}
.glyphicon-text-width:before {
  content: "\e051";
}
.glyphicon-align-left:before {
  content: "\e052";
}
.glyphicon-align-center:before {
  content: "\e053";
}
.glyphicon-align-right:before {
  content: "\e054";
}
.glyphicon-align-justify:before {
  content: "\e055";
}
.glyphicon-list:before {
  content: "\e056";
}
.glyphicon-indent-left:before {
  content: "\e057";
}
.glyphicon-indent-right:before {
  content: "\e058";
}
.glyphicon-facetime-video:before {
  content: "\e059";
}
.glyphicon-picture:before {
  content: "\e060";
}
.glyphicon-map-marker:before {
  content: "\e062";
}
.glyphicon-adjust:before {
  content: "\e063";
}
.glyphicon-tint:before {
  content: "\e064";
}
.glyphicon-edit:before {
  content: "\e065";
}
.glyphicon-share:before {
  content: "\e066";
}
.glyphicon-check:before {
  content: "\e067";
}
.glyphicon-move:before {
  content: "\e068";
}
.glyphicon-step-backward:before {
  content: "\e069";
}
.glyphicon-fast-backward:before {
  content: "\e070";
}
.glyphicon-backward:before {
  content: "\e071";
}
.glyphicon-play:before {
  content: "\e072";
}
.glyphicon-pause:before {
  content: "\e073";
}
.glyphicon-stop:before {
  content: "\e074";
}
.glyphicon-forward:before {
  content: "\e075";
}
.glyphicon-fast-forward:before {
  content: "\e076";
}
.glyphicon-step-forward:before {
  content: "\e077";
}
.glyphicon-eject:before {
  content: "\e078";
}
.glyphicon-chevron-left:before {
  content: "\e079";
}
.glyphicon-chevron-right:before {
  content: "\e080";
}
.glyphicon-plus-sign:before {
  content: "\e081";
}
.glyphicon-minus-sign:before {
  content: "\e082";
}
.glyphicon-remove-sign:before {
  content: "\e083";
}
.glyphicon-ok-sign:before {
  content: "\e084";
}
.glyphicon-question-sign:before {
  content: "\e085";
}
.glyphicon-info-sign:before {
  content: "\e086";
}
.glyphicon-screenshot:before {
  content: "\e087";
}
.glyphicon-remove-circle:before {
  content: "\e088";
}
.glyphicon-ok-circle:before {
  content: "\e089";
}
.glyphicon-ban-circle:before {
  content: "\e090";
}
.glyphicon-arrow-left:before {
  content: "\e091";
}
.glyphicon-arrow-right:before {
  content: "\e092";
}
.glyphicon-arrow-up:before {
  content: "\e093";
}
.glyphicon-arrow-down:before {
  content: "\e094";
}
.glyphicon-share-alt:before {
  content: "\e095";
}
.glyphicon-resize-full:before {
  content: "\e096";
}
.glyphicon-resize-small:before {
  content: "\e097";
}
.glyphicon-exclamation-sign:before {
  content: "\e101";
}
.glyphicon-gift:before {
  content: "\e102";
}
.glyphicon-leaf:before {
  content: "\e103";
}
.glyphicon-fire:before {
  content: "\e104";
}
.glyphicon-eye-open:before {
  content: "\e105";
}
.glyphicon-eye-close:before {
  content: "\e106";
}
.glyphicon-warning-sign:before {
  content: "\e107";
}
.glyphicon-plane:before {
  content: "\e108";
}
.glyphicon-calendar:before {
  content: "\e109";
}
.glyphicon-random:before {
  content: "\e110";
}
.glyphicon-comment:before {
  content: "\e111";
}
.glyphicon-magnet:before {
  content: "\e112";
}
.glyphicon-chevron-up:before {
  content: "\e113";
}
.glyphicon-chevron-down:before {
  content: "\e114";
}
.glyphicon-retweet:before {
  content: "\e115";
}
.glyphicon-shopping-cart:before {
  content: "\e116";
}
.glyphicon-folder-close:before {
  content: "\e117";
}
.glyphicon-folder-open:before {
  content: "\e118";
}
.glyphicon-resize-vertical:before {
  content: "\e119";
}
.glyphicon-resize-horizontal:before {
  content: "\e120";
}
.glyphicon-hdd:before {
  content: "\e121";
}
.glyphicon-bullhorn:before {
  content: "\e122";
}
.glyphicon-bell:before {
  content: "\e123";
}
.glyphicon-certificate:before {
  content: "\e124";
}
.glyphicon-thumbs-up:before {
  content: "\e125";
}
.glyphicon-thumbs-down:before {
  content: "\e126";
}
.glyphicon-hand-right:before {
  content: "\e127";
}
.glyphicon-hand-left:before {
  content: "\e128";
}
.glyphicon-hand-up:before {
  content: "\e129";
}
.glyphicon-hand-down:before {
  content: "\e130";
}
.glyphicon-circle-arrow-right:before {
  content: "\e131";
}
.glyphicon-circle-arrow-left:before {
  content: "\e132";
}
.glyphicon-circle-arrow-up:before {
  content: "\e133";
}
.glyphicon-circle-arrow-down:before {
  content: "\e134";
}
.glyphicon-globe:before {
  content: "\e135";
}
.glyphicon-wrench:before {
  content: "\e136";
}
.glyphicon-tasks:before {
  content: "\e137";
}
.glyphicon-filter:before {
  content: "\e138";
}
.glyphicon-briefcase:before {
  content: "\e139";
}
.glyphicon-fullscreen:before {
  content: "\e140";
}
.glyphicon-dashboard:before {
  content: "\e141";
}
.glyphicon-paperclip:before {
  content: "\e142";
}
.glyphicon-heart-empty:before {
  content: "\e143";
}
.glyphicon-link:before {
  content: "\e144";
}
.glyphicon-phone:before {
  content: "\e145";
}
.glyphicon-pushpin:before {
  content: "\e146";
}
.glyphicon-usd:before {
  content: "\e148";
}
.glyphicon-gbp:before {
  content: "\e149";
}
.glyphicon-sort:before {
  content: "\e150";
}
.glyphicon-sort-by-alphabet:before {
  content: "\e151";
}
.glyphicon-sort-by-alphabet-alt:before {
  content: "\e152";
}
.glyphicon-sort-by-order:before {
  content: "\e153";
}
.glyphicon-sort-by-order-alt:before {
  content: "\e154";
}
.glyphicon-sort-by-attributes:before {
  content: "\e155";
}
.glyphicon-sort-by-attributes-alt:before {
  content: "\e156";
}
.glyphicon-unchecked:before {
  content: "\e157";
}
.glyphicon-expand:before {
  content: "\e158";
}
.glyphicon-collapse-down:before {
  content: "\e159";
}
.glyphicon-collapse-up:before {
  content: "\e160";
}
.glyphicon-log-in:before {
  content: "\e161";
}
.glyphicon-flash:before {
  content: "\e162";
}
.glyphicon-log-out:before {
  content: "\e163";
}
.glyphicon-new-window:before {
  content: "\e164";
}
.glyphicon-record:before {
  content: "\e165";
}
.glyphicon-save:before {
  content: "\e166";
}
.glyphicon-open:before {
  content: "\e167";
}
.glyphicon-saved:before {
  content: "\e168";
}
.glyphicon-import:before {
  content: "\e169";
}
.glyphicon-export:before {
  content: "\e170";
}
.glyphicon-send:before {
  content: "\e171";
}
.glyphicon-floppy-disk:before {
  content: "\e172";
}
.glyphicon-floppy-saved:before {
  content: "\e173";
}
.glyphicon-floppy-remove:before {
  content: "\e174";
}
.glyphicon-floppy-save:before {
  content: "\e175";
}
.glyphicon-floppy-open:before {
  content: "\e176";
}
.glyphicon-credit-card:before {
  content: "\e177";
}
.glyphicon-transfer:before {
  content: "\e178";
}
.glyphicon-cutlery:before {
  content: "\e179";
}
.glyphicon-header:before {
  content: "\e180";
}
.glyphicon-compressed:before {
  content: "\e181";
}
.glyphicon-earphone:before {
  content: "\e182";
}
.glyphicon-phone-alt:before {
  content: "\e183";
}
.glyphicon-tower:before {
  content: "\e184";
}
.glyphicon-stats:before {
  content: "\e185";
}
.glyphicon-sd-video:before {
  content: "\e186";
}
.glyphicon-hd-video:before {
  content: "\e187";
}
.glyphicon-subtitles:before {
  content: "\e188";
}
.glyphicon-sound-stereo:before {
  content: "\e189";
}
.glyphicon-sound-dolby:before {
  content: "\e190";
}
.glyphicon-sound-5-1:before {
  content: "\e191";
}
.glyphicon-sound-6-1:before {
  content: "\e192";
}
.glyphicon-sound-7-1:before {
  content: "\e193";
}
.glyphicon-copyright-mark:before {
  content: "\e194";
}
.glyphicon-registration-mark:before {
  content: "\e195";
}
.glyphicon-cloud-download:before {
  content: "\e197";
}
.glyphicon-cloud-upload:before {
  content: "\e198";
}
.glyphicon-tree-conifer:before {
  content: "\e199";
}
.glyphicon-tree-deciduous:before {
  content: "\e200";
}
.glyphicon-cd:before {
  content: "\e201";
}
.glyphicon-save-file:before {
  content: "\e202";
}
.glyphicon-open-file:before {
  content: "\e203";
}
.glyphicon-level-up:before {
  content: "\e204";
}
.glyphicon-copy:before {
  content: "\e205";
}
.glyphicon-paste:before {
  content: "\e206";
}
.glyphicon-alert:before {
  content: "\e209";
}
.glyphicon-equalizer:before {
  content: "\e210";
}
.glyphicon-king:before {
  content: "\e211";
}
.glyphicon-queen:before {
  content: "\e212";
}
.glyphicon-pawn:before {
  content: "\e213";
}
.glyphicon-bishop:before {
  content: "\e214";
}
.glyphicon-knight:before {
  content: "\e215";
}
.glyphicon-baby-formula:before {
  content: "\e216";
}
.glyphicon-tent:before {
  content: "\26fa";
}
.glyphicon-blackboard:before {
  content: "\e218";
}
.glyphicon-bed:before {
  content: "\e219";
}
.glyphicon-apple:before {
  content: "\f8ff";
}
.glyphicon-erase:before {
  content: "\e221";
}
.glyphicon-hourglass:before {
  content: "\231b";
}
.glyphicon-lamp:before {
  content: "\e223";
}
.glyphicon-duplicate:before {
  content: "\e224";
}
.glyphicon-piggy-bank:before {
  content: "\e225";
}
.glyphicon-scissors:before {
  content: "\e226";
}
.glyphicon-bitcoin:before {
  content: "\e227";
}
.glyphicon-btc:before {
  content: "\e227";
}
.glyphicon-xbt:before {
  content: "\e227";
}
.glyphicon-yen:before {
  content: "\00a5";
}
.glyphicon-jpy:before {
  content: "\00a5";
}
.glyphicon-ruble:before {
  content: "\20bd";
}
.glyphicon-rub:before {
  content: "\20bd";
}
.glyphicon-scale:before {
  content: "\e230";
}
.glyphicon-ice-lolly:before {
  content: "\e231";
}
.glyphicon-ice-lolly-tasted:before {
  content: "\e232";
}
.glyphicon-education:before {
  content: "\e233";
}
.glyphicon-option-horizontal:before {
  content: "\e234";
}
.glyphicon-option-vertical:before {
  content: "\e235";
}
.glyphicon-menu-hamburger:before {
  content: "\e236";
}
.glyphicon-modal-window:before {
  content: "\e237";
}
.glyphicon-oil:before {
  content: "\e238";
}
.glyphicon-grain:before {
  content: "\e239";
}
.glyphicon-sunglasses:before {
  content: "\e240";
}
.glyphicon-text-size:before {
  content: "\e241";
}
.glyphicon-text-color:before {
  content: "\e242";
}
.glyphicon-text-background:before {
  content: "\e243";
}
.glyphicon-object-align-top:before {
  content: "\e244";
}
.glyphicon-object-align-bottom:before {
  content: "\e245";
}
.glyphicon-object-align-horizontal:before {
  content: "\e246";
}
.glyphicon-object-align-left:before {
  content: "\e247";
}
.glyphicon-object-align-vertical:before {
  content: "\e248";
}
.glyphicon-object-align-right:before {
  content: "\e249";
}
.glyphicon-triangle-right:before {
  content: "\e250";
}
.glyphicon-triangle-left:before {
  content: "\e251";
}
.glyphicon-triangle-bottom:before {
  content: "\e252";
}
.glyphicon-triangle-top:before {
  content: "\e253";
}
.glyphicon-console:before {
  content: "\e254";
}
.glyphicon-superscript:before {
  content: "\e255";
}
.glyphicon-subscript:before {
  content: "\e256";
}
.glyphicon-menu-left:before {
  content: "\e257";
}
.glyphicon-menu-right:before {
  content: "\e258";
}
.glyphicon-menu-down:before {
  content: "\e259";
}
.glyphicon-menu-up:before {
  content: "\e260";
}
* {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
*:before,
*:after {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
html {
  font-size: 10px;
  -webkit-tap-highlight-color: rgba(0, 0, 0, 0);
}
body {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 13px;
  line-height: 1.42857143;
  color: #000;
  background-color: #fff;
}
input,
button,
select,
textarea {
  font-family: inherit;
  font-size: inherit;
  line-height: inherit;
}
a {
  color: #337ab7;
  text-decoration: none;
}
a:hover,
a:focus {
  color: #23527c;
  text-decoration: underline;
}
a:focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
figure {
  margin: 0;
}
img {
  vertical-align: middle;
}
.img-responsive,
.thumbnail > img,
.thumbnail a > img,
.carousel-inner > .item > img,
.carousel-inner > .item > a > img {
  display: block;
  max-width: 100%;
  height: auto;
}
.img-rounded {
  border-radius: 3px;
}
.img-thumbnail {
  padding: 4px;
  line-height: 1.42857143;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 2px;
  -webkit-transition: all 0.2s ease-in-out;
  -o-transition: all 0.2s ease-in-out;
  transition: all 0.2s ease-in-out;
  display: inline-block;
  max-width: 100%;
  height: auto;
}
.img-circle {
  border-radius: 50%;
}
hr {
  margin-top: 18px;
  margin-bottom: 18px;
  border: 0;
  border-top: 1px solid #eeeeee;
}
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  margin: -1px;
  padding: 0;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}
.sr-only-focusable:active,
.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  margin: 0;
  overflow: visible;
  clip: auto;
}
[role="button"] {
  cursor: pointer;
}
h1,
h2,
h3,
h4,
h5,
h6,
.h1,
.h2,
.h3,
.h4,
.h5,
.h6 {
  font-family: inherit;
  font-weight: 500;
  line-height: 1.1;
  color: inherit;
}
h1 small,
h2 small,
h3 small,
h4 small,
h5 small,
h6 small,
.h1 small,
.h2 small,
.h3 small,
.h4 small,
.h5 small,
.h6 small,
h1 .small,
h2 .small,
h3 .small,
h4 .small,
h5 .small,
h6 .small,
.h1 .small,
.h2 .small,
.h3 .small,
.h4 .small,
.h5 .small,
.h6 .small {
  font-weight: normal;
  line-height: 1;
  color: #777777;
}
h1,
.h1,
h2,
.h2,
h3,
.h3 {
  margin-top: 18px;
  margin-bottom: 9px;
}
h1 small,
.h1 small,
h2 small,
.h2 small,
h3 small,
.h3 small,
h1 .small,
.h1 .small,
h2 .small,
.h2 .small,
h3 .small,
.h3 .small {
  font-size: 65%;
}
h4,
.h4,
h5,
.h5,
h6,
.h6 {
  margin-top: 9px;
  margin-bottom: 9px;
}
h4 small,
.h4 small,
h5 small,
.h5 small,
h6 small,
.h6 small,
h4 .small,
.h4 .small,
h5 .small,
.h5 .small,
h6 .small,
.h6 .small {
  font-size: 75%;
}
h1,
.h1 {
  font-size: 33px;
}
h2,
.h2 {
  font-size: 27px;
}
h3,
.h3 {
  font-size: 23px;
}
h4,
.h4 {
  font-size: 17px;
}
h5,
.h5 {
  font-size: 13px;
}
h6,
.h6 {
  font-size: 12px;
}
p {
  margin: 0 0 9px;
}
.lead {
  margin-bottom: 18px;
  font-size: 14px;
  font-weight: 300;
  line-height: 1.4;
}
@media (min-width: 768px) {
  .lead {
    font-size: 19.5px;
  }
}
small,
.small {
  font-size: 92%;
}
mark,
.mark {
  background-color: #fcf8e3;
  padding: .2em;
}
.text-left {
  text-align: left;
}
.text-right {
  text-align: right;
}
.text-center {
  text-align: center;
}
.text-justify {
  text-align: justify;
}
.text-nowrap {
  white-space: nowrap;
}
.text-lowercase {
  text-transform: lowercase;
}
.text-uppercase {
  text-transform: uppercase;
}
.text-capitalize {
  text-transform: capitalize;
}
.text-muted {
  color: #777777;
}
.text-primary {
  color: #337ab7;
}
a.text-primary:hover,
a.text-primary:focus {
  color: #286090;
}
.text-success {
  color: #3c763d;
}
a.text-success:hover,
a.text-success:focus {
  color: #2b542c;
}
.text-info {
  color: #31708f;
}
a.text-info:hover,
a.text-info:focus {
  color: #245269;
}
.text-warning {
  color: #8a6d3b;
}
a.text-warning:hover,
a.text-warning:focus {
  color: #66512c;
}
.text-danger {
  color: #a94442;
}
a.text-danger:hover,
a.text-danger:focus {
  color: #843534;
}
.bg-primary {
  color: #fff;
  background-color: #337ab7;
}
a.bg-primary:hover,
a.bg-primary:focus {
  background-color: #286090;
}
.bg-success {
  background-color: #dff0d8;
}
a.bg-success:hover,
a.bg-success:focus {
  background-color: #c1e2b3;
}
.bg-info {
  background-color: #d9edf7;
}
a.bg-info:hover,
a.bg-info:focus {
  background-color: #afd9ee;
}
.bg-warning {
  background-color: #fcf8e3;
}
a.bg-warning:hover,
a.bg-warning:focus {
  background-color: #f7ecb5;
}
.bg-danger {
  background-color: #f2dede;
}
a.bg-danger:hover,
a.bg-danger:focus {
  background-color: #e4b9b9;
}
.page-header {
  padding-bottom: 8px;
  margin: 36px 0 18px;
  border-bottom: 1px solid #eeeeee;
}
ul,
ol {
  margin-top: 0;
  margin-bottom: 9px;
}
ul ul,
ol ul,
ul ol,
ol ol {
  margin-bottom: 0;
}
.list-unstyled {
  padding-left: 0;
  list-style: none;
}
.list-inline {
  padding-left: 0;
  list-style: none;
  margin-left: -5px;
}
.list-inline > li {
  display: inline-block;
  padding-left: 5px;
  padding-right: 5px;
}
dl {
  margin-top: 0;
  margin-bottom: 18px;
}
dt,
dd {
  line-height: 1.42857143;
}
dt {
  font-weight: bold;
}
dd {
  margin-left: 0;
}
@media (min-width: 541px) {
  .dl-horizontal dt {
    float: left;
    width: 160px;
    clear: left;
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .dl-horizontal dd {
    margin-left: 180px;
  }
}
abbr[title],
abbr[data-original-title] {
  cursor: help;
  border-bottom: 1px dotted #777777;
}
.initialism {
  font-size: 90%;
  text-transform: uppercase;
}
blockquote {
  padding: 9px 18px;
  margin: 0 0 18px;
  font-size: inherit;
  border-left: 5px solid #eeeeee;
}
blockquote p:last-child,
blockquote ul:last-child,
blockquote ol:last-child {
  margin-bottom: 0;
}
blockquote footer,
blockquote small,
blockquote .small {
  display: block;
  font-size: 80%;
  line-height: 1.42857143;
  color: #777777;
}
blockquote footer:before,
blockquote small:before,
blockquote .small:before {
  content: '\2014 \00A0';
}
.blockquote-reverse,
blockquote.pull-right {
  padding-right: 15px;
  padding-left: 0;
  border-right: 5px solid #eeeeee;
  border-left: 0;
  text-align: right;
}
.blockquote-reverse footer:before,
blockquote.pull-right footer:before,
.blockquote-reverse small:before,
blockquote.pull-right small:before,
.blockquote-reverse .small:before,
blockquote.pull-right .small:before {
  content: '';
}
.blockquote-reverse footer:after,
blockquote.pull-right footer:after,
.blockquote-reverse small:after,
blockquote.pull-right small:after,
.blockquote-reverse .small:after,
blockquote.pull-right .small:after {
  content: '\00A0 \2014';
}
address {
  margin-bottom: 18px;
  font-style: normal;
  line-height: 1.42857143;
}
code,
kbd,
pre,
samp {
  font-family: monospace;
}
code {
  padding: 2px 4px;
  font-size: 90%;
  color: #c7254e;
  background-color: #f9f2f4;
  border-radius: 2px;
}
kbd {
  padding: 2px 4px;
  font-size: 90%;
  color: #888;
  background-color: transparent;
  border-radius: 1px;
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.25);
}
kbd kbd {
  padding: 0;
  font-size: 100%;
  font-weight: bold;
  box-shadow: none;
}
pre {
  display: block;
  padding: 8.5px;
  margin: 0 0 9px;
  font-size: 12px;
  line-height: 1.42857143;
  word-break: break-all;
  word-wrap: break-word;
  color: #333333;
  background-color: #f5f5f5;
  border: 1px solid #ccc;
  border-radius: 2px;
}
pre code {
  padding: 0;
  font-size: inherit;
  color: inherit;
  white-space: pre-wrap;
  background-color: transparent;
  border-radius: 0;
}
.pre-scrollable {
  max-height: 340px;
  overflow-y: scroll;
}
.container {
  margin-right: auto;
  margin-left: auto;
  padding-left: 0px;
  padding-right: 0px;
}
@media (min-width: 768px) {
  .container {
    width: 768px;
  }
}
@media (min-width: 992px) {
  .container {
    width: 940px;
  }
}
@media (min-width: 1200px) {
  .container {
    width: 1140px;
  }
}
.container-fluid {
  margin-right: auto;
  margin-left: auto;
  padding-left: 0px;
  padding-right: 0px;
}
.row {
  margin-left: 0px;
  margin-right: 0px;
}
.col-xs-1, .col-sm-1, .col-md-1, .col-lg-1, .col-xs-2, .col-sm-2, .col-md-2, .col-lg-2, .col-xs-3, .col-sm-3, .col-md-3, .col-lg-3, .col-xs-4, .col-sm-4, .col-md-4, .col-lg-4, .col-xs-5, .col-sm-5, .col-md-5, .col-lg-5, .col-xs-6, .col-sm-6, .col-md-6, .col-lg-6, .col-xs-7, .col-sm-7, .col-md-7, .col-lg-7, .col-xs-8, .col-sm-8, .col-md-8, .col-lg-8, .col-xs-9, .col-sm-9, .col-md-9, .col-lg-9, .col-xs-10, .col-sm-10, .col-md-10, .col-lg-10, .col-xs-11, .col-sm-11, .col-md-11, .col-lg-11, .col-xs-12, .col-sm-12, .col-md-12, .col-lg-12 {
  position: relative;
  min-height: 1px;
  padding-left: 0px;
  padding-right: 0px;
}
.col-xs-1, .col-xs-2, .col-xs-3, .col-xs-4, .col-xs-5, .col-xs-6, .col-xs-7, .col-xs-8, .col-xs-9, .col-xs-10, .col-xs-11, .col-xs-12 {
  float: left;
}
.col-xs-12 {
  width: 100%;
}
.col-xs-11 {
  width: 91.66666667%;
}
.col-xs-10 {
  width: 83.33333333%;
}
.col-xs-9 {
  width: 75%;
}
.col-xs-8 {
  width: 66.66666667%;
}
.col-xs-7 {
  width: 58.33333333%;
}
.col-xs-6 {
  width: 50%;
}
.col-xs-5 {
  width: 41.66666667%;
}
.col-xs-4 {
  width: 33.33333333%;
}
.col-xs-3 {
  width: 25%;
}
.col-xs-2 {
  width: 16.66666667%;
}
.col-xs-1 {
  width: 8.33333333%;
}
.col-xs-pull-12 {
  right: 100%;
}
.col-xs-pull-11 {
  right: 91.66666667%;
}
.col-xs-pull-10 {
  right: 83.33333333%;
}
.col-xs-pull-9 {
  right: 75%;
}
.col-xs-pull-8 {
  right: 66.66666667%;
}
.col-xs-pull-7 {
  right: 58.33333333%;
}
.col-xs-pull-6 {
  right: 50%;
}
.col-xs-pull-5 {
  right: 41.66666667%;
}
.col-xs-pull-4 {
  right: 33.33333333%;
}
.col-xs-pull-3 {
  right: 25%;
}
.col-xs-pull-2 {
  right: 16.66666667%;
}
.col-xs-pull-1 {
  right: 8.33333333%;
}
.col-xs-pull-0 {
  right: auto;
}
.col-xs-push-12 {
  left: 100%;
}
.col-xs-push-11 {
  left: 91.66666667%;
}
.col-xs-push-10 {
  left: 83.33333333%;
}
.col-xs-push-9 {
  left: 75%;
}
.col-xs-push-8 {
  left: 66.66666667%;
}
.col-xs-push-7 {
  left: 58.33333333%;
}
.col-xs-push-6 {
  left: 50%;
}
.col-xs-push-5 {
  left: 41.66666667%;
}
.col-xs-push-4 {
  left: 33.33333333%;
}
.col-xs-push-3 {
  left: 25%;
}
.col-xs-push-2 {
  left: 16.66666667%;
}
.col-xs-push-1 {
  left: 8.33333333%;
}
.col-xs-push-0 {
  left: auto;
}
.col-xs-offset-12 {
  margin-left: 100%;
}
.col-xs-offset-11 {
  margin-left: 91.66666667%;
}
.col-xs-offset-10 {
  margin-left: 83.33333333%;
}
.col-xs-offset-9 {
  margin-left: 75%;
}
.col-xs-offset-8 {
  margin-left: 66.66666667%;
}
.col-xs-offset-7 {
  margin-left: 58.33333333%;
}
.col-xs-offset-6 {
  margin-left: 50%;
}
.col-xs-offset-5 {
  margin-left: 41.66666667%;
}
.col-xs-offset-4 {
  margin-left: 33.33333333%;
}
.col-xs-offset-3 {
  margin-left: 25%;
}
.col-xs-offset-2 {
  margin-left: 16.66666667%;
}
.col-xs-offset-1 {
  margin-left: 8.33333333%;
}
.col-xs-offset-0 {
  margin-left: 0%;
}
@media (min-width: 768px) {
  .col-sm-1, .col-sm-2, .col-sm-3, .col-sm-4, .col-sm-5, .col-sm-6, .col-sm-7, .col-sm-8, .col-sm-9, .col-sm-10, .col-sm-11, .col-sm-12 {
    float: left;
  }
  .col-sm-12 {
    width: 100%;
  }
  .col-sm-11 {
    width: 91.66666667%;
  }
  .col-sm-10 {
    width: 83.33333333%;
  }
  .col-sm-9 {
    width: 75%;
  }
  .col-sm-8 {
    width: 66.66666667%;
  }
  .col-sm-7 {
    width: 58.33333333%;
  }
  .col-sm-6 {
    width: 50%;
  }
  .col-sm-5 {
    width: 41.66666667%;
  }
  .col-sm-4 {
    width: 33.33333333%;
  }
  .col-sm-3 {
    width: 25%;
  }
  .col-sm-2 {
    width: 16.66666667%;
  }
  .col-sm-1 {
    width: 8.33333333%;
  }
  .col-sm-pull-12 {
    right: 100%;
  }
  .col-sm-pull-11 {
    right: 91.66666667%;
  }
  .col-sm-pull-10 {
    right: 83.33333333%;
  }
  .col-sm-pull-9 {
    right: 75%;
  }
  .col-sm-pull-8 {
    right: 66.66666667%;
  }
  .col-sm-pull-7 {
    right: 58.33333333%;
  }
  .col-sm-pull-6 {
    right: 50%;
  }
  .col-sm-pull-5 {
    right: 41.66666667%;
  }
  .col-sm-pull-4 {
    right: 33.33333333%;
  }
  .col-sm-pull-3 {
    right: 25%;
  }
  .col-sm-pull-2 {
    right: 16.66666667%;
  }
  .col-sm-pull-1 {
    right: 8.33333333%;
  }
  .col-sm-pull-0 {
    right: auto;
  }
  .col-sm-push-12 {
    left: 100%;
  }
  .col-sm-push-11 {
    left: 91.66666667%;
  }
  .col-sm-push-10 {
    left: 83.33333333%;
  }
  .col-sm-push-9 {
    left: 75%;
  }
  .col-sm-push-8 {
    left: 66.66666667%;
  }
  .col-sm-push-7 {
    left: 58.33333333%;
  }
  .col-sm-push-6 {
    left: 50%;
  }
  .col-sm-push-5 {
    left: 41.66666667%;
  }
  .col-sm-push-4 {
    left: 33.33333333%;
  }
  .col-sm-push-3 {
    left: 25%;
  }
  .col-sm-push-2 {
    left: 16.66666667%;
  }
  .col-sm-push-1 {
    left: 8.33333333%;
  }
  .col-sm-push-0 {
    left: auto;
  }
  .col-sm-offset-12 {
    margin-left: 100%;
  }
  .col-sm-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-sm-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-sm-offset-9 {
    margin-left: 75%;
  }
  .col-sm-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-sm-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-sm-offset-6 {
    margin-left: 50%;
  }
  .col-sm-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-sm-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-sm-offset-3 {
    margin-left: 25%;
  }
  .col-sm-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-sm-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-sm-offset-0 {
    margin-left: 0%;
  }
}
@media (min-width: 992px) {
  .col-md-1, .col-md-2, .col-md-3, .col-md-4, .col-md-5, .col-md-6, .col-md-7, .col-md-8, .col-md-9, .col-md-10, .col-md-11, .col-md-12 {
    float: left;
  }
  .col-md-12 {
    width: 100%;
  }
  .col-md-11 {
    width: 91.66666667%;
  }
  .col-md-10 {
    width: 83.33333333%;
  }
  .col-md-9 {
    width: 75%;
  }
  .col-md-8 {
    width: 66.66666667%;
  }
  .col-md-7 {
    width: 58.33333333%;
  }
  .col-md-6 {
    width: 50%;
  }
  .col-md-5 {
    width: 41.66666667%;
  }
  .col-md-4 {
    width: 33.33333333%;
  }
  .col-md-3 {
    width: 25%;
  }
  .col-md-2 {
    width: 16.66666667%;
  }
  .col-md-1 {
    width: 8.33333333%;
  }
  .col-md-pull-12 {
    right: 100%;
  }
  .col-md-pull-11 {
    right: 91.66666667%;
  }
  .col-md-pull-10 {
    right: 83.33333333%;
  }
  .col-md-pull-9 {
    right: 75%;
  }
  .col-md-pull-8 {
    right: 66.66666667%;
  }
  .col-md-pull-7 {
    right: 58.33333333%;
  }
  .col-md-pull-6 {
    right: 50%;
  }
  .col-md-pull-5 {
    right: 41.66666667%;
  }
  .col-md-pull-4 {
    right: 33.33333333%;
  }
  .col-md-pull-3 {
    right: 25%;
  }
  .col-md-pull-2 {
    right: 16.66666667%;
  }
  .col-md-pull-1 {
    right: 8.33333333%;
  }
  .col-md-pull-0 {
    right: auto;
  }
  .col-md-push-12 {
    left: 100%;
  }
  .col-md-push-11 {
    left: 91.66666667%;
  }
  .col-md-push-10 {
    left: 83.33333333%;
  }
  .col-md-push-9 {
    left: 75%;
  }
  .col-md-push-8 {
    left: 66.66666667%;
  }
  .col-md-push-7 {
    left: 58.33333333%;
  }
  .col-md-push-6 {
    left: 50%;
  }
  .col-md-push-5 {
    left: 41.66666667%;
  }
  .col-md-push-4 {
    left: 33.33333333%;
  }
  .col-md-push-3 {
    left: 25%;
  }
  .col-md-push-2 {
    left: 16.66666667%;
  }
  .col-md-push-1 {
    left: 8.33333333%;
  }
  .col-md-push-0 {
    left: auto;
  }
  .col-md-offset-12 {
    margin-left: 100%;
  }
  .col-md-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-md-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-md-offset-9 {
    margin-left: 75%;
  }
  .col-md-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-md-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-md-offset-6 {
    margin-left: 50%;
  }
  .col-md-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-md-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-md-offset-3 {
    margin-left: 25%;
  }
  .col-md-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-md-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-md-offset-0 {
    margin-left: 0%;
  }
}
@media (min-width: 1200px) {
  .col-lg-1, .col-lg-2, .col-lg-3, .col-lg-4, .col-lg-5, .col-lg-6, .col-lg-7, .col-lg-8, .col-lg-9, .col-lg-10, .col-lg-11, .col-lg-12 {
    float: left;
  }
  .col-lg-12 {
    width: 100%;
  }
  .col-lg-11 {
    width: 91.66666667%;
  }
  .col-lg-10 {
    width: 83.33333333%;
  }
  .col-lg-9 {
    width: 75%;
  }
  .col-lg-8 {
    width: 66.66666667%;
  }
  .col-lg-7 {
    width: 58.33333333%;
  }
  .col-lg-6 {
    width: 50%;
  }
  .col-lg-5 {
    width: 41.66666667%;
  }
  .col-lg-4 {
    width: 33.33333333%;
  }
  .col-lg-3 {
    width: 25%;
  }
  .col-lg-2 {
    width: 16.66666667%;
  }
  .col-lg-1 {
    width: 8.33333333%;
  }
  .col-lg-pull-12 {
    right: 100%;
  }
  .col-lg-pull-11 {
    right: 91.66666667%;
  }
  .col-lg-pull-10 {
    right: 83.33333333%;
  }
  .col-lg-pull-9 {
    right: 75%;
  }
  .col-lg-pull-8 {
    right: 66.66666667%;
  }
  .col-lg-pull-7 {
    right: 58.33333333%;
  }
  .col-lg-pull-6 {
    right: 50%;
  }
  .col-lg-pull-5 {
    right: 41.66666667%;
  }
  .col-lg-pull-4 {
    right: 33.33333333%;
  }
  .col-lg-pull-3 {
    right: 25%;
  }
  .col-lg-pull-2 {
    right: 16.66666667%;
  }
  .col-lg-pull-1 {
    right: 8.33333333%;
  }
  .col-lg-pull-0 {
    right: auto;
  }
  .col-lg-push-12 {
    left: 100%;
  }
  .col-lg-push-11 {
    left: 91.66666667%;
  }
  .col-lg-push-10 {
    left: 83.33333333%;
  }
  .col-lg-push-9 {
    left: 75%;
  }
  .col-lg-push-8 {
    left: 66.66666667%;
  }
  .col-lg-push-7 {
    left: 58.33333333%;
  }
  .col-lg-push-6 {
    left: 50%;
  }
  .col-lg-push-5 {
    left: 41.66666667%;
  }
  .col-lg-push-4 {
    left: 33.33333333%;
  }
  .col-lg-push-3 {
    left: 25%;
  }
  .col-lg-push-2 {
    left: 16.66666667%;
  }
  .col-lg-push-1 {
    left: 8.33333333%;
  }
  .col-lg-push-0 {
    left: auto;
  }
  .col-lg-offset-12 {
    margin-left: 100%;
  }
  .col-lg-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-lg-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-lg-offset-9 {
    margin-left: 75%;
  }
  .col-lg-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-lg-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-lg-offset-6 {
    margin-left: 50%;
  }
  .col-lg-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-lg-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-lg-offset-3 {
    margin-left: 25%;
  }
  .col-lg-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-lg-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-lg-offset-0 {
    margin-left: 0%;
  }
}
table {
  background-color: transparent;
}
caption {
  padding-top: 8px;
  padding-bottom: 8px;
  color: #777777;
  text-align: left;
}
th {
  text-align: left;
}
.table {
  width: 100%;
  max-width: 100%;
  margin-bottom: 18px;
}
.table > thead > tr > th,
.table > tbody > tr > th,
.table > tfoot > tr > th,
.table > thead > tr > td,
.table > tbody > tr > td,
.table > tfoot > tr > td {
  padding: 8px;
  line-height: 1.42857143;
  vertical-align: top;
  border-top: 1px solid #ddd;
}
.table > thead > tr > th {
  vertical-align: bottom;
  border-bottom: 2px solid #ddd;
}
.table > caption + thead > tr:first-child > th,
.table > colgroup + thead > tr:first-child > th,
.table > thead:first-child > tr:first-child > th,
.table > caption + thead > tr:first-child > td,
.table > colgroup + thead > tr:first-child > td,
.table > thead:first-child > tr:first-child > td {
  border-top: 0;
}
.table > tbody + tbody {
  border-top: 2px solid #ddd;
}
.table .table {
  background-color: #fff;
}
.table-condensed > thead > tr > th,
.table-condensed > tbody > tr > th,
.table-condensed > tfoot > tr > th,
.table-condensed > thead > tr > td,
.table-condensed > tbody > tr > td,
.table-condensed > tfoot > tr > td {
  padding: 5px;
}
.table-bordered {
  border: 1px solid #ddd;
}
.table-bordered > thead > tr > th,
.table-bordered > tbody > tr > th,
.table-bordered > tfoot > tr > th,
.table-bordered > thead > tr > td,
.table-bordered > tbody > tr > td,
.table-bordered > tfoot > tr > td {
  border: 1px solid #ddd;
}
.table-bordered > thead > tr > th,
.table-bordered > thead > tr > td {
  border-bottom-width: 2px;
}
.table-striped > tbody > tr:nth-of-type(odd) {
  background-color: #f9f9f9;
}
.table-hover > tbody > tr:hover {
  background-color: #f5f5f5;
}
table col[class*="col-"] {
  position: static;
  float: none;
  display: table-column;
}
table td[class*="col-"],
table th[class*="col-"] {
  position: static;
  float: none;
  display: table-cell;
}
.table > thead > tr > td.active,
.table > tbody > tr > td.active,
.table > tfoot > tr > td.active,
.table > thead > tr > th.active,
.table > tbody > tr > th.active,
.table > tfoot > tr > th.active,
.table > thead > tr.active > td,
.table > tbody > tr.active > td,
.table > tfoot > tr.active > td,
.table > thead > tr.active > th,
.table > tbody > tr.active > th,
.table > tfoot > tr.active > th {
  background-color: #f5f5f5;
}
.table-hover > tbody > tr > td.active:hover,
.table-hover > tbody > tr > th.active:hover,
.table-hover > tbody > tr.active:hover > td,
.table-hover > tbody > tr:hover > .active,
.table-hover > tbody > tr.active:hover > th {
  background-color: #e8e8e8;
}
.table > thead > tr > td.success,
.table > tbody > tr > td.success,
.table > tfoot > tr > td.success,
.table > thead > tr > th.success,
.table > tbody > tr > th.success,
.table > tfoot > tr > th.success,
.table > thead > tr.success > td,
.table > tbody > tr.success > td,
.table > tfoot > tr.success > td,
.table > thead > tr.success > th,
.table > tbody > tr.success > th,
.table > tfoot > tr.success > th {
  background-color: #dff0d8;
}
.table-hover > tbody > tr > td.success:hover,
.table-hover > tbody > tr > th.success:hover,
.table-hover > tbody > tr.success:hover > td,
.table-hover > tbody > tr:hover > .success,
.table-hover > tbody > tr.success:hover > th {
  background-color: #d0e9c6;
}
.table > thead > tr > td.info,
.table > tbody > tr > td.info,
.table > tfoot > tr > td.info,
.table > thead > tr > th.info,
.table > tbody > tr > th.info,
.table > tfoot > tr > th.info,
.table > thead > tr.info > td,
.table > tbody > tr.info > td,
.table > tfoot > tr.info > td,
.table > thead > tr.info > th,
.table > tbody > tr.info > th,
.table > tfoot > tr.info > th {
  background-color: #d9edf7;
}
.table-hover > tbody > tr > td.info:hover,
.table-hover > tbody > tr > th.info:hover,
.table-hover > tbody > tr.info:hover > td,
.table-hover > tbody > tr:hover > .info,
.table-hover > tbody > tr.info:hover > th {
  background-color: #c4e3f3;
}
.table > thead > tr > td.warning,
.table > tbody > tr > td.warning,
.table > tfoot > tr > td.warning,
.table > thead > tr > th.warning,
.table > tbody > tr > th.warning,
.table > tfoot > tr > th.warning,
.table > thead > tr.warning > td,
.table > tbody > tr.warning > td,
.table > tfoot > tr.warning > td,
.table > thead > tr.warning > th,
.table > tbody > tr.warning > th,
.table > tfoot > tr.warning > th {
  background-color: #fcf8e3;
}
.table-hover > tbody > tr > td.warning:hover,
.table-hover > tbody > tr > th.warning:hover,
.table-hover > tbody > tr.warning:hover > td,
.table-hover > tbody > tr:hover > .warning,
.table-hover > tbody > tr.warning:hover > th {
  background-color: #faf2cc;
}
.table > thead > tr > td.danger,
.table > tbody > tr > td.danger,
.table > tfoot > tr > td.danger,
.table > thead > tr > th.danger,
.table > tbody > tr > th.danger,
.table > tfoot > tr > th.danger,
.table > thead > tr.danger > td,
.table > tbody > tr.danger > td,
.table > tfoot > tr.danger > td,
.table > thead > tr.danger > th,
.table > tbody > tr.danger > th,
.table > tfoot > tr.danger > th {
  background-color: #f2dede;
}
.table-hover > tbody > tr > td.danger:hover,
.table-hover > tbody > tr > th.danger:hover,
.table-hover > tbody > tr.danger:hover > td,
.table-hover > tbody > tr:hover > .danger,
.table-hover > tbody > tr.danger:hover > th {
  background-color: #ebcccc;
}
.table-responsive {
  overflow-x: auto;
  min-height: 0.01%;
}
@media screen and (max-width: 767px) {
  .table-responsive {
    width: 100%;
    margin-bottom: 13.5px;
    overflow-y: hidden;
    -ms-overflow-style: -ms-autohiding-scrollbar;
    border: 1px solid #ddd;
  }
  .table-responsive > .table {
    margin-bottom: 0;
  }
  .table-responsive > .table > thead > tr > th,
  .table-responsive > .table > tbody > tr > th,
  .table-responsive > .table > tfoot > tr > th,
  .table-responsive > .table > thead > tr > td,
  .table-responsive > .table > tbody > tr > td,
  .table-responsive > .table > tfoot > tr > td {
    white-space: nowrap;
  }
  .table-responsive > .table-bordered {
    border: 0;
  }
  .table-responsive > .table-bordered > thead > tr > th:first-child,
  .table-responsive > .table-bordered > tbody > tr > th:first-child,
  .table-responsive > .table-bordered > tfoot > tr > th:first-child,
  .table-responsive > .table-bordered > thead > tr > td:first-child,
  .table-responsive > .table-bordered > tbody > tr > td:first-child,
  .table-responsive > .table-bordered > tfoot > tr > td:first-child {
    border-left: 0;
  }
  .table-responsive > .table-bordered > thead > tr > th:last-child,
  .table-responsive > .table-bordered > tbody > tr > th:last-child,
  .table-responsive > .table-bordered > tfoot > tr > th:last-child,
  .table-responsive > .table-bordered > thead > tr > td:last-child,
  .table-responsive > .table-bordered > tbody > tr > td:last-child,
  .table-responsive > .table-bordered > tfoot > tr > td:last-child {
    border-right: 0;
  }
  .table-responsive > .table-bordered > tbody > tr:last-child > th,
  .table-responsive > .table-bordered > tfoot > tr:last-child > th,
  .table-responsive > .table-bordered > tbody > tr:last-child > td,
  .table-responsive > .table-bordered > tfoot > tr:last-child > td {
    border-bottom: 0;
  }
}
fieldset {
  padding: 0;
  margin: 0;
  border: 0;
  min-width: 0;
}
legend {
  display: block;
  width: 100%;
  padding: 0;
  margin-bottom: 18px;
  font-size: 19.5px;
  line-height: inherit;
  color: #333333;
  border: 0;
  border-bottom: 1px solid #e5e5e5;
}
label {
  display: inline-block;
  max-width: 100%;
  margin-bottom: 5px;
  font-weight: bold;
}
input[type="search"] {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
input[type="radio"],
input[type="checkbox"] {
  margin: 4px 0 0;
  margin-top: 1px \9;
  line-height: normal;
}
input[type="file"] {
  display: block;
}
input[type="range"] {
  display: block;
  width: 100%;
}
select[multiple],
select[size] {
  height: auto;
}
input[type="file"]:focus,
input[type="radio"]:focus,
input[type="checkbox"]:focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
output {
  display: block;
  padding-top: 7px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
}
.form-control {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
}
.form-control:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.form-control::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.form-control:-ms-input-placeholder {
  color: #999;
}
.form-control::-webkit-input-placeholder {
  color: #999;
}
.form-control::-ms-expand {
  border: 0;
  background-color: transparent;
}
.form-control[disabled],
.form-control[readonly],
fieldset[disabled] .form-control {
  background-color: #eeeeee;
  opacity: 1;
}
.form-control[disabled],
fieldset[disabled] .form-control {
  cursor: not-allowed;
}
textarea.form-control {
  height: auto;
}
input[type="search"] {
  -webkit-appearance: none;
}
@media screen and (-webkit-min-device-pixel-ratio: 0) {
  input[type="date"].form-control,
  input[type="time"].form-control,
  input[type="datetime-local"].form-control,
  input[type="month"].form-control {
    line-height: 32px;
  }
  input[type="date"].input-sm,
  input[type="time"].input-sm,
  input[type="datetime-local"].input-sm,
  input[type="month"].input-sm,
  .input-group-sm input[type="date"],
  .input-group-sm input[type="time"],
  .input-group-sm input[type="datetime-local"],
  .input-group-sm input[type="month"] {
    line-height: 30px;
  }
  input[type="date"].input-lg,
  input[type="time"].input-lg,
  input[type="datetime-local"].input-lg,
  input[type="month"].input-lg,
  .input-group-lg input[type="date"],
  .input-group-lg input[type="time"],
  .input-group-lg input[type="datetime-local"],
  .input-group-lg input[type="month"] {
    line-height: 45px;
  }
}
.form-group {
  margin-bottom: 15px;
}
.radio,
.checkbox {
  position: relative;
  display: block;
  margin-top: 10px;
  margin-bottom: 10px;
}
.radio label,
.checkbox label {
  min-height: 18px;
  padding-left: 20px;
  margin-bottom: 0;
  font-weight: normal;
  cursor: pointer;
}
.radio input[type="radio"],
.radio-inline input[type="radio"],
.checkbox input[type="checkbox"],
.checkbox-inline input[type="checkbox"] {
  position: absolute;
  margin-left: -20px;
  margin-top: 4px \9;
}
.radio + .radio,
.checkbox + .checkbox {
  margin-top: -5px;
}
.radio-inline,
.checkbox-inline {
  position: relative;
  display: inline-block;
  padding-left: 20px;
  margin-bottom: 0;
  vertical-align: middle;
  font-weight: normal;
  cursor: pointer;
}
.radio-inline + .radio-inline,
.checkbox-inline + .checkbox-inline {
  margin-top: 0;
  margin-left: 10px;
}
input[type="radio"][disabled],
input[type="checkbox"][disabled],
input[type="radio"].disabled,
input[type="checkbox"].disabled,
fieldset[disabled] input[type="radio"],
fieldset[disabled] input[type="checkbox"] {
  cursor: not-allowed;
}
.radio-inline.disabled,
.checkbox-inline.disabled,
fieldset[disabled] .radio-inline,
fieldset[disabled] .checkbox-inline {
  cursor: not-allowed;
}
.radio.disabled label,
.checkbox.disabled label,
fieldset[disabled] .radio label,
fieldset[disabled] .checkbox label {
  cursor: not-allowed;
}
.form-control-static {
  padding-top: 7px;
  padding-bottom: 7px;
  margin-bottom: 0;
  min-height: 31px;
}
.form-control-static.input-lg,
.form-control-static.input-sm {
  padding-left: 0;
  padding-right: 0;
}
.input-sm {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
select.input-sm {
  height: 30px;
  line-height: 30px;
}
textarea.input-sm,
select[multiple].input-sm {
  height: auto;
}
.form-group-sm .form-control {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.form-group-sm select.form-control {
  height: 30px;
  line-height: 30px;
}
.form-group-sm textarea.form-control,
.form-group-sm select[multiple].form-control {
  height: auto;
}
.form-group-sm .form-control-static {
  height: 30px;
  min-height: 30px;
  padding: 6px 10px;
  font-size: 12px;
  line-height: 1.5;
}
.input-lg {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
select.input-lg {
  height: 45px;
  line-height: 45px;
}
textarea.input-lg,
select[multiple].input-lg {
  height: auto;
}
.form-group-lg .form-control {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
.form-group-lg select.form-control {
  height: 45px;
  line-height: 45px;
}
.form-group-lg textarea.form-control,
.form-group-lg select[multiple].form-control {
  height: auto;
}
.form-group-lg .form-control-static {
  height: 45px;
  min-height: 35px;
  padding: 11px 16px;
  font-size: 17px;
  line-height: 1.3333333;
}
.has-feedback {
  position: relative;
}
.has-feedback .form-control {
  padding-right: 40px;
}
.form-control-feedback {
  position: absolute;
  top: 0;
  right: 0;
  z-index: 2;
  display: block;
  width: 32px;
  height: 32px;
  line-height: 32px;
  text-align: center;
  pointer-events: none;
}
.input-lg + .form-control-feedback,
.input-group-lg + .form-control-feedback,
.form-group-lg .form-control + .form-control-feedback {
  width: 45px;
  height: 45px;
  line-height: 45px;
}
.input-sm + .form-control-feedback,
.input-group-sm + .form-control-feedback,
.form-group-sm .form-control + .form-control-feedback {
  width: 30px;
  height: 30px;
  line-height: 30px;
}
.has-success .help-block,
.has-success .control-label,
.has-success .radio,
.has-success .checkbox,
.has-success .radio-inline,
.has-success .checkbox-inline,
.has-success.radio label,
.has-success.checkbox label,
.has-success.radio-inline label,
.has-success.checkbox-inline label {
  color: #3c763d;
}
.has-success .form-control {
  border-color: #3c763d;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-success .form-control:focus {
  border-color: #2b542c;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #67b168;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #67b168;
}
.has-success .input-group-addon {
  color: #3c763d;
  border-color: #3c763d;
  background-color: #dff0d8;
}
.has-success .form-control-feedback {
  color: #3c763d;
}
.has-warning .help-block,
.has-warning .control-label,
.has-warning .radio,
.has-warning .checkbox,
.has-warning .radio-inline,
.has-warning .checkbox-inline,
.has-warning.radio label,
.has-warning.checkbox label,
.has-warning.radio-inline label,
.has-warning.checkbox-inline label {
  color: #8a6d3b;
}
.has-warning .form-control {
  border-color: #8a6d3b;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-warning .form-control:focus {
  border-color: #66512c;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #c0a16b;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #c0a16b;
}
.has-warning .input-group-addon {
  color: #8a6d3b;
  border-color: #8a6d3b;
  background-color: #fcf8e3;
}
.has-warning .form-control-feedback {
  color: #8a6d3b;
}
.has-error .help-block,
.has-error .control-label,
.has-error .radio,
.has-error .checkbox,
.has-error .radio-inline,
.has-error .checkbox-inline,
.has-error.radio label,
.has-error.checkbox label,
.has-error.radio-inline label,
.has-error.checkbox-inline label {
  color: #a94442;
}
.has-error .form-control {
  border-color: #a94442;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-error .form-control:focus {
  border-color: #843534;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #ce8483;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #ce8483;
}
.has-error .input-group-addon {
  color: #a94442;
  border-color: #a94442;
  background-color: #f2dede;
}
.has-error .form-control-feedback {
  color: #a94442;
}
.has-feedback label ~ .form-control-feedback {
  top: 23px;
}
.has-feedback label.sr-only ~ .form-control-feedback {
  top: 0;
}
.help-block {
  display: block;
  margin-top: 5px;
  margin-bottom: 10px;
  color: #404040;
}
@media (min-width: 768px) {
  .form-inline .form-group {
    display: inline-block;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .form-control {
    display: inline-block;
    width: auto;
    vertical-align: middle;
  }
  .form-inline .form-control-static {
    display: inline-block;
  }
  .form-inline .input-group {
    display: inline-table;
    vertical-align: middle;
  }
  .form-inline .input-group .input-group-addon,
  .form-inline .input-group .input-group-btn,
  .form-inline .input-group .form-control {
    width: auto;
  }
  .form-inline .input-group > .form-control {
    width: 100%;
  }
  .form-inline .control-label {
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .radio,
  .form-inline .checkbox {
    display: inline-block;
    margin-top: 0;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .radio label,
  .form-inline .checkbox label {
    padding-left: 0;
  }
  .form-inline .radio input[type="radio"],
  .form-inline .checkbox input[type="checkbox"] {
    position: relative;
    margin-left: 0;
  }
  .form-inline .has-feedback .form-control-feedback {
    top: 0;
  }
}
.form-horizontal .radio,
.form-horizontal .checkbox,
.form-horizontal .radio-inline,
.form-horizontal .checkbox-inline {
  margin-top: 0;
  margin-bottom: 0;
  padding-top: 7px;
}
.form-horizontal .radio,
.form-horizontal .checkbox {
  min-height: 25px;
}
.form-horizontal .form-group {
  margin-left: 0px;
  margin-right: 0px;
}
@media (min-width: 768px) {
  .form-horizontal .control-label {
    text-align: right;
    margin-bottom: 0;
    padding-top: 7px;
  }
}
.form-horizontal .has-feedback .form-control-feedback {
  right: 0px;
}
@media (min-width: 768px) {
  .form-horizontal .form-group-lg .control-label {
    padding-top: 11px;
    font-size: 17px;
  }
}
@media (min-width: 768px) {
  .form-horizontal .form-group-sm .control-label {
    padding-top: 6px;
    font-size: 12px;
  }
}
.btn {
  display: inline-block;
  margin-bottom: 0;
  font-weight: normal;
  text-align: center;
  vertical-align: middle;
  touch-action: manipulation;
  cursor: pointer;
  background-image: none;
  border: 1px solid transparent;
  white-space: nowrap;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  border-radius: 2px;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}
.btn:focus,
.btn:active:focus,
.btn.active:focus,
.btn.focus,
.btn:active.focus,
.btn.active.focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
.btn:hover,
.btn:focus,
.btn.focus {
  color: #333;
  text-decoration: none;
}
.btn:active,
.btn.active {
  outline: 0;
  background-image: none;
  -webkit-box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
  box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
}
.btn.disabled,
.btn[disabled],
fieldset[disabled] .btn {
  cursor: not-allowed;
  opacity: 0.65;
  filter: alpha(opacity=65);
  -webkit-box-shadow: none;
  box-shadow: none;
}
a.btn.disabled,
fieldset[disabled] a.btn {
  pointer-events: none;
}
.btn-default {
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
.btn-default:focus,
.btn-default.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
.btn-default:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.btn-default:active,
.btn-default.active,
.open > .dropdown-toggle.btn-default {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.btn-default:active:hover,
.btn-default.active:hover,
.open > .dropdown-toggle.btn-default:hover,
.btn-default:active:focus,
.btn-default.active:focus,
.open > .dropdown-toggle.btn-default:focus,
.btn-default:active.focus,
.btn-default.active.focus,
.open > .dropdown-toggle.btn-default.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
.btn-default:active,
.btn-default.active,
.open > .dropdown-toggle.btn-default {
  background-image: none;
}
.btn-default.disabled:hover,
.btn-default[disabled]:hover,
fieldset[disabled] .btn-default:hover,
.btn-default.disabled:focus,
.btn-default[disabled]:focus,
fieldset[disabled] .btn-default:focus,
.btn-default.disabled.focus,
.btn-default[disabled].focus,
fieldset[disabled] .btn-default.focus {
  background-color: #fff;
  border-color: #ccc;
}
.btn-default .badge {
  color: #fff;
  background-color: #333;
}
.btn-primary {
  color: #fff;
  background-color: #337ab7;
  border-color: #2e6da4;
}
.btn-primary:focus,
.btn-primary.focus {
  color: #fff;
  background-color: #286090;
  border-color: #122b40;
}
.btn-primary:hover {
  color: #fff;
  background-color: #286090;
  border-color: #204d74;
}
.btn-primary:active,
.btn-primary.active,
.open > .dropdown-toggle.btn-primary {
  color: #fff;
  background-color: #286090;
  border-color: #204d74;
}
.btn-primary:active:hover,
.btn-primary.active:hover,
.open > .dropdown-toggle.btn-primary:hover,
.btn-primary:active:focus,
.btn-primary.active:focus,
.open > .dropdown-toggle.btn-primary:focus,
.btn-primary:active.focus,
.btn-primary.active.focus,
.open > .dropdown-toggle.btn-primary.focus {
  color: #fff;
  background-color: #204d74;
  border-color: #122b40;
}
.btn-primary:active,
.btn-primary.active,
.open > .dropdown-toggle.btn-primary {
  background-image: none;
}
.btn-primary.disabled:hover,
.btn-primary[disabled]:hover,
fieldset[disabled] .btn-primary:hover,
.btn-primary.disabled:focus,
.btn-primary[disabled]:focus,
fieldset[disabled] .btn-primary:focus,
.btn-primary.disabled.focus,
.btn-primary[disabled].focus,
fieldset[disabled] .btn-primary.focus {
  background-color: #337ab7;
  border-color: #2e6da4;
}
.btn-primary .badge {
  color: #337ab7;
  background-color: #fff;
}
.btn-success {
  color: #fff;
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.btn-success:focus,
.btn-success.focus {
  color: #fff;
  background-color: #449d44;
  border-color: #255625;
}
.btn-success:hover {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.btn-success:active,
.btn-success.active,
.open > .dropdown-toggle.btn-success {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.btn-success:active:hover,
.btn-success.active:hover,
.open > .dropdown-toggle.btn-success:hover,
.btn-success:active:focus,
.btn-success.active:focus,
.open > .dropdown-toggle.btn-success:focus,
.btn-success:active.focus,
.btn-success.active.focus,
.open > .dropdown-toggle.btn-success.focus {
  color: #fff;
  background-color: #398439;
  border-color: #255625;
}
.btn-success:active,
.btn-success.active,
.open > .dropdown-toggle.btn-success {
  background-image: none;
}
.btn-success.disabled:hover,
.btn-success[disabled]:hover,
fieldset[disabled] .btn-success:hover,
.btn-success.disabled:focus,
.btn-success[disabled]:focus,
fieldset[disabled] .btn-success:focus,
.btn-success.disabled.focus,
.btn-success[disabled].focus,
fieldset[disabled] .btn-success.focus {
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.btn-success .badge {
  color: #5cb85c;
  background-color: #fff;
}
.btn-info {
  color: #fff;
  background-color: #5bc0de;
  border-color: #46b8da;
}
.btn-info:focus,
.btn-info.focus {
  color: #fff;
  background-color: #31b0d5;
  border-color: #1b6d85;
}
.btn-info:hover {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.btn-info:active,
.btn-info.active,
.open > .dropdown-toggle.btn-info {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.btn-info:active:hover,
.btn-info.active:hover,
.open > .dropdown-toggle.btn-info:hover,
.btn-info:active:focus,
.btn-info.active:focus,
.open > .dropdown-toggle.btn-info:focus,
.btn-info:active.focus,
.btn-info.active.focus,
.open > .dropdown-toggle.btn-info.focus {
  color: #fff;
  background-color: #269abc;
  border-color: #1b6d85;
}
.btn-info:active,
.btn-info.active,
.open > .dropdown-toggle.btn-info {
  background-image: none;
}
.btn-info.disabled:hover,
.btn-info[disabled]:hover,
fieldset[disabled] .btn-info:hover,
.btn-info.disabled:focus,
.btn-info[disabled]:focus,
fieldset[disabled] .btn-info:focus,
.btn-info.disabled.focus,
.btn-info[disabled].focus,
fieldset[disabled] .btn-info.focus {
  background-color: #5bc0de;
  border-color: #46b8da;
}
.btn-info .badge {
  color: #5bc0de;
  background-color: #fff;
}
.btn-warning {
  color: #fff;
  background-color: #f0ad4e;
  border-color: #eea236;
}
.btn-warning:focus,
.btn-warning.focus {
  color: #fff;
  background-color: #ec971f;
  border-color: #985f0d;
}
.btn-warning:hover {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.btn-warning:active,
.btn-warning.active,
.open > .dropdown-toggle.btn-warning {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.btn-warning:active:hover,
.btn-warning.active:hover,
.open > .dropdown-toggle.btn-warning:hover,
.btn-warning:active:focus,
.btn-warning.active:focus,
.open > .dropdown-toggle.btn-warning:focus,
.btn-warning:active.focus,
.btn-warning.active.focus,
.open > .dropdown-toggle.btn-warning.focus {
  color: #fff;
  background-color: #d58512;
  border-color: #985f0d;
}
.btn-warning:active,
.btn-warning.active,
.open > .dropdown-toggle.btn-warning {
  background-image: none;
}
.btn-warning.disabled:hover,
.btn-warning[disabled]:hover,
fieldset[disabled] .btn-warning:hover,
.btn-warning.disabled:focus,
.btn-warning[disabled]:focus,
fieldset[disabled] .btn-warning:focus,
.btn-warning.disabled.focus,
.btn-warning[disabled].focus,
fieldset[disabled] .btn-warning.focus {
  background-color: #f0ad4e;
  border-color: #eea236;
}
.btn-warning .badge {
  color: #f0ad4e;
  background-color: #fff;
}
.btn-danger {
  color: #fff;
  background-color: #d9534f;
  border-color: #d43f3a;
}
.btn-danger:focus,
.btn-danger.focus {
  color: #fff;
  background-color: #c9302c;
  border-color: #761c19;
}
.btn-danger:hover {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.btn-danger:active,
.btn-danger.active,
.open > .dropdown-toggle.btn-danger {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.btn-danger:active:hover,
.btn-danger.active:hover,
.open > .dropdown-toggle.btn-danger:hover,
.btn-danger:active:focus,
.btn-danger.active:focus,
.open > .dropdown-toggle.btn-danger:focus,
.btn-danger:active.focus,
.btn-danger.active.focus,
.open > .dropdown-toggle.btn-danger.focus {
  color: #fff;
  background-color: #ac2925;
  border-color: #761c19;
}
.btn-danger:active,
.btn-danger.active,
.open > .dropdown-toggle.btn-danger {
  background-image: none;
}
.btn-danger.disabled:hover,
.btn-danger[disabled]:hover,
fieldset[disabled] .btn-danger:hover,
.btn-danger.disabled:focus,
.btn-danger[disabled]:focus,
fieldset[disabled] .btn-danger:focus,
.btn-danger.disabled.focus,
.btn-danger[disabled].focus,
fieldset[disabled] .btn-danger.focus {
  background-color: #d9534f;
  border-color: #d43f3a;
}
.btn-danger .badge {
  color: #d9534f;
  background-color: #fff;
}
.btn-link {
  color: #337ab7;
  font-weight: normal;
  border-radius: 0;
}
.btn-link,
.btn-link:active,
.btn-link.active,
.btn-link[disabled],
fieldset[disabled] .btn-link {
  background-color: transparent;
  -webkit-box-shadow: none;
  box-shadow: none;
}
.btn-link,
.btn-link:hover,
.btn-link:focus,
.btn-link:active {
  border-color: transparent;
}
.btn-link:hover,
.btn-link:focus {
  color: #23527c;
  text-decoration: underline;
  background-color: transparent;
}
.btn-link[disabled]:hover,
fieldset[disabled] .btn-link:hover,
.btn-link[disabled]:focus,
fieldset[disabled] .btn-link:focus {
  color: #777777;
  text-decoration: none;
}
.btn-lg,
.btn-group-lg > .btn {
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
.btn-sm,
.btn-group-sm > .btn {
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.btn-xs,
.btn-group-xs > .btn {
  padding: 1px 5px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.btn-block {
  display: block;
  width: 100%;
}
.btn-block + .btn-block {
  margin-top: 5px;
}
input[type="submit"].btn-block,
input[type="reset"].btn-block,
input[type="button"].btn-block {
  width: 100%;
}
.fade {
  opacity: 0;
  -webkit-transition: opacity 0.15s linear;
  -o-transition: opacity 0.15s linear;
  transition: opacity 0.15s linear;
}
.fade.in {
  opacity: 1;
}
.collapse {
  display: none;
}
.collapse.in {
  display: block;
}
tr.collapse.in {
  display: table-row;
}
tbody.collapse.in {
  display: table-row-group;
}
.collapsing {
  position: relative;
  height: 0;
  overflow: hidden;
  -webkit-transition-property: height, visibility;
  transition-property: height, visibility;
  -webkit-transition-duration: 0.35s;
  transition-duration: 0.35s;
  -webkit-transition-timing-function: ease;
  transition-timing-function: ease;
}
.caret {
  display: inline-block;
  width: 0;
  height: 0;
  margin-left: 2px;
  vertical-align: middle;
  border-top: 4px dashed;
  border-top: 4px solid \9;
  border-right: 4px solid transparent;
  border-left: 4px solid transparent;
}
.dropup,
.dropdown {
  position: relative;
}
.dropdown-toggle:focus {
  outline: 0;
}
.dropdown-menu {
  position: absolute;
  top: 100%;
  left: 0;
  z-index: 1000;
  display: none;
  float: left;
  min-width: 160px;
  padding: 5px 0;
  margin: 2px 0 0;
  list-style: none;
  font-size: 13px;
  text-align: left;
  background-color: #fff;
  border: 1px solid #ccc;
  border: 1px solid rgba(0, 0, 0, 0.15);
  border-radius: 2px;
  -webkit-box-shadow: 0 6px 12px rgba(0, 0, 0, 0.175);
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.175);
  background-clip: padding-box;
}
.dropdown-menu.pull-right {
  right: 0;
  left: auto;
}
.dropdown-menu .divider {
  height: 1px;
  margin: 8px 0;
  overflow: hidden;
  background-color: #e5e5e5;
}
.dropdown-menu > li > a {
  display: block;
  padding: 3px 20px;
  clear: both;
  font-weight: normal;
  line-height: 1.42857143;
  color: #333333;
  white-space: nowrap;
}
.dropdown-menu > li > a:hover,
.dropdown-menu > li > a:focus {
  text-decoration: none;
  color: #262626;
  background-color: #f5f5f5;
}
.dropdown-menu > .active > a,
.dropdown-menu > .active > a:hover,
.dropdown-menu > .active > a:focus {
  color: #fff;
  text-decoration: none;
  outline: 0;
  background-color: #337ab7;
}
.dropdown-menu > .disabled > a,
.dropdown-menu > .disabled > a:hover,
.dropdown-menu > .disabled > a:focus {
  color: #777777;
}
.dropdown-menu > .disabled > a:hover,
.dropdown-menu > .disabled > a:focus {
  text-decoration: none;
  background-color: transparent;
  background-image: none;
  filter: progid:DXImageTransform.Microsoft.gradient(enabled = false);
  cursor: not-allowed;
}
.open > .dropdown-menu {
  display: block;
}
.open > a {
  outline: 0;
}
.dropdown-menu-right {
  left: auto;
  right: 0;
}
.dropdown-menu-left {
  left: 0;
  right: auto;
}
.dropdown-header {
  display: block;
  padding: 3px 20px;
  font-size: 12px;
  line-height: 1.42857143;
  color: #777777;
  white-space: nowrap;
}
.dropdown-backdrop {
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;
  top: 0;
  z-index: 990;
}
.pull-right > .dropdown-menu {
  right: 0;
  left: auto;
}
.dropup .caret,
.navbar-fixed-bottom .dropdown .caret {
  border-top: 0;
  border-bottom: 4px dashed;
  border-bottom: 4px solid \9;
  content: "";
}
.dropup .dropdown-menu,
.navbar-fixed-bottom .dropdown .dropdown-menu {
  top: auto;
  bottom: 100%;
  margin-bottom: 2px;
}
@media (min-width: 541px) {
  .navbar-right .dropdown-menu {
    left: auto;
    right: 0;
  }
  .navbar-right .dropdown-menu-left {
    left: 0;
    right: auto;
  }
}
.btn-group,
.btn-group-vertical {
  position: relative;
  display: inline-block;
  vertical-align: middle;
}
.btn-group > .btn,
.btn-group-vertical > .btn {
  position: relative;
  float: left;
}
.btn-group > .btn:hover,
.btn-group-vertical > .btn:hover,
.btn-group > .btn:focus,
.btn-group-vertical > .btn:focus,
.btn-group > .btn:active,
.btn-group-vertical > .btn:active,
.btn-group > .btn.active,
.btn-group-vertical > .btn.active {
  z-index: 2;
}
.btn-group .btn + .btn,
.btn-group .btn + .btn-group,
.btn-group .btn-group + .btn,
.btn-group .btn-group + .btn-group {
  margin-left: -1px;
}
.btn-toolbar {
  margin-left: -5px;
}
.btn-toolbar .btn,
.btn-toolbar .btn-group,
.btn-toolbar .input-group {
  float: left;
}
.btn-toolbar > .btn,
.btn-toolbar > .btn-group,
.btn-toolbar > .input-group {
  margin-left: 5px;
}
.btn-group > .btn:not(:first-child):not(:last-child):not(.dropdown-toggle) {
  border-radius: 0;
}
.btn-group > .btn:first-child {
  margin-left: 0;
}
.btn-group > .btn:first-child:not(:last-child):not(.dropdown-toggle) {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.btn-group > .btn:last-child:not(:first-child),
.btn-group > .dropdown-toggle:not(:first-child) {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.btn-group > .btn-group {
  float: left;
}
.btn-group > .btn-group:not(:first-child):not(:last-child) > .btn {
  border-radius: 0;
}
.btn-group > .btn-group:first-child:not(:last-child) > .btn:last-child,
.btn-group > .btn-group:first-child:not(:last-child) > .dropdown-toggle {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.btn-group > .btn-group:last-child:not(:first-child) > .btn:first-child {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.btn-group .dropdown-toggle:active,
.btn-group.open .dropdown-toggle {
  outline: 0;
}
.btn-group > .btn + .dropdown-toggle {
  padding-left: 8px;
  padding-right: 8px;
}
.btn-group > .btn-lg + .dropdown-toggle {
  padding-left: 12px;
  padding-right: 12px;
}
.btn-group.open .dropdown-toggle {
  -webkit-box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
  box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
}
.btn-group.open .dropdown-toggle.btn-link {
  -webkit-box-shadow: none;
  box-shadow: none;
}
.btn .caret {
  margin-left: 0;
}
.btn-lg .caret {
  border-width: 5px 5px 0;
  border-bottom-width: 0;
}
.dropup .btn-lg .caret {
  border-width: 0 5px 5px;
}
.btn-group-vertical > .btn,
.btn-group-vertical > .btn-group,
.btn-group-vertical > .btn-group > .btn {
  display: block;
  float: none;
  width: 100%;
  max-width: 100%;
}
.btn-group-vertical > .btn-group > .btn {
  float: none;
}
.btn-group-vertical > .btn + .btn,
.btn-group-vertical > .btn + .btn-group,
.btn-group-vertical > .btn-group + .btn,
.btn-group-vertical > .btn-group + .btn-group {
  margin-top: -1px;
  margin-left: 0;
}
.btn-group-vertical > .btn:not(:first-child):not(:last-child) {
  border-radius: 0;
}
.btn-group-vertical > .btn:first-child:not(:last-child) {
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.btn-group-vertical > .btn:last-child:not(:first-child) {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
  border-bottom-right-radius: 2px;
  border-bottom-left-radius: 2px;
}
.btn-group-vertical > .btn-group:not(:first-child):not(:last-child) > .btn {
  border-radius: 0;
}
.btn-group-vertical > .btn-group:first-child:not(:last-child) > .btn:last-child,
.btn-group-vertical > .btn-group:first-child:not(:last-child) > .dropdown-toggle {
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.btn-group-vertical > .btn-group:last-child:not(:first-child) > .btn:first-child {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.btn-group-justified {
  display: table;
  width: 100%;
  table-layout: fixed;
  border-collapse: separate;
}
.btn-group-justified > .btn,
.btn-group-justified > .btn-group {
  float: none;
  display: table-cell;
  width: 1%;
}
.btn-group-justified > .btn-group .btn {
  width: 100%;
}
.btn-group-justified > .btn-group .dropdown-menu {
  left: auto;
}
[data-toggle="buttons"] > .btn input[type="radio"],
[data-toggle="buttons"] > .btn-group > .btn input[type="radio"],
[data-toggle="buttons"] > .btn input[type="checkbox"],
[data-toggle="buttons"] > .btn-group > .btn input[type="checkbox"] {
  position: absolute;
  clip: rect(0, 0, 0, 0);
  pointer-events: none;
}
.input-group {
  position: relative;
  display: table;
  border-collapse: separate;
}
.input-group[class*="col-"] {
  float: none;
  padding-left: 0;
  padding-right: 0;
}
.input-group .form-control {
  position: relative;
  z-index: 2;
  float: left;
  width: 100%;
  margin-bottom: 0;
}
.input-group .form-control:focus {
  z-index: 3;
}
.input-group-lg > .form-control,
.input-group-lg > .input-group-addon,
.input-group-lg > .input-group-btn > .btn {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
select.input-group-lg > .form-control,
select.input-group-lg > .input-group-addon,
select.input-group-lg > .input-group-btn > .btn {
  height: 45px;
  line-height: 45px;
}
textarea.input-group-lg > .form-control,
textarea.input-group-lg > .input-group-addon,
textarea.input-group-lg > .input-group-btn > .btn,
select[multiple].input-group-lg > .form-control,
select[multiple].input-group-lg > .input-group-addon,
select[multiple].input-group-lg > .input-group-btn > .btn {
  height: auto;
}
.input-group-sm > .form-control,
.input-group-sm > .input-group-addon,
.input-group-sm > .input-group-btn > .btn {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
select.input-group-sm > .form-control,
select.input-group-sm > .input-group-addon,
select.input-group-sm > .input-group-btn > .btn {
  height: 30px;
  line-height: 30px;
}
textarea.input-group-sm > .form-control,
textarea.input-group-sm > .input-group-addon,
textarea.input-group-sm > .input-group-btn > .btn,
select[multiple].input-group-sm > .form-control,
select[multiple].input-group-sm > .input-group-addon,
select[multiple].input-group-sm > .input-group-btn > .btn {
  height: auto;
}
.input-group-addon,
.input-group-btn,
.input-group .form-control {
  display: table-cell;
}
.input-group-addon:not(:first-child):not(:last-child),
.input-group-btn:not(:first-child):not(:last-child),
.input-group .form-control:not(:first-child):not(:last-child) {
  border-radius: 0;
}
.input-group-addon,
.input-group-btn {
  width: 1%;
  white-space: nowrap;
  vertical-align: middle;
}
.input-group-addon {
  padding: 6px 12px;
  font-size: 13px;
  font-weight: normal;
  line-height: 1;
  color: #555555;
  text-align: center;
  background-color: #eeeeee;
  border: 1px solid #ccc;
  border-radius: 2px;
}
.input-group-addon.input-sm {
  padding: 5px 10px;
  font-size: 12px;
  border-radius: 1px;
}
.input-group-addon.input-lg {
  padding: 10px 16px;
  font-size: 17px;
  border-radius: 3px;
}
.input-group-addon input[type="radio"],
.input-group-addon input[type="checkbox"] {
  margin-top: 0;
}
.input-group .form-control:first-child,
.input-group-addon:first-child,
.input-group-btn:first-child > .btn,
.input-group-btn:first-child > .btn-group > .btn,
.input-group-btn:first-child > .dropdown-toggle,
.input-group-btn:last-child > .btn:not(:last-child):not(.dropdown-toggle),
.input-group-btn:last-child > .btn-group:not(:last-child) > .btn {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.input-group-addon:first-child {
  border-right: 0;
}
.input-group .form-control:last-child,
.input-group-addon:last-child,
.input-group-btn:last-child > .btn,
.input-group-btn:last-child > .btn-group > .btn,
.input-group-btn:last-child > .dropdown-toggle,
.input-group-btn:first-child > .btn:not(:first-child),
.input-group-btn:first-child > .btn-group:not(:first-child) > .btn {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.input-group-addon:last-child {
  border-left: 0;
}
.input-group-btn {
  position: relative;
  font-size: 0;
  white-space: nowrap;
}
.input-group-btn > .btn {
  position: relative;
}
.input-group-btn > .btn + .btn {
  margin-left: -1px;
}
.input-group-btn > .btn:hover,
.input-group-btn > .btn:focus,
.input-group-btn > .btn:active {
  z-index: 2;
}
.input-group-btn:first-child > .btn,
.input-group-btn:first-child > .btn-group {
  margin-right: -1px;
}
.input-group-btn:last-child > .btn,
.input-group-btn:last-child > .btn-group {
  z-index: 2;
  margin-left: -1px;
}
.nav {
  margin-bottom: 0;
  padding-left: 0;
  list-style: none;
}
.nav > li {
  position: relative;
  display: block;
}
.nav > li > a {
  position: relative;
  display: block;
  padding: 10px 15px;
}
.nav > li > a:hover,
.nav > li > a:focus {
  text-decoration: none;
  background-color: #eeeeee;
}
.nav > li.disabled > a {
  color: #777777;
}
.nav > li.disabled > a:hover,
.nav > li.disabled > a:focus {
  color: #777777;
  text-decoration: none;
  background-color: transparent;
  cursor: not-allowed;
}
.nav .open > a,
.nav .open > a:hover,
.nav .open > a:focus {
  background-color: #eeeeee;
  border-color: #337ab7;
}
.nav .nav-divider {
  height: 1px;
  margin: 8px 0;
  overflow: hidden;
  background-color: #e5e5e5;
}
.nav > li > a > img {
  max-width: none;
}
.nav-tabs {
  border-bottom: 1px solid #ddd;
}
.nav-tabs > li {
  float: left;
  margin-bottom: -1px;
}
.nav-tabs > li > a {
  margin-right: 2px;
  line-height: 1.42857143;
  border: 1px solid transparent;
  border-radius: 2px 2px 0 0;
}
.nav-tabs > li > a:hover {
  border-color: #eeeeee #eeeeee #ddd;
}
.nav-tabs > li.active > a,
.nav-tabs > li.active > a:hover,
.nav-tabs > li.active > a:focus {
  color: #555555;
  background-color: #fff;
  border: 1px solid #ddd;
  border-bottom-color: transparent;
  cursor: default;
}
.nav-tabs.nav-justified {
  width: 100%;
  border-bottom: 0;
}
.nav-tabs.nav-justified > li {
  float: none;
}
.nav-tabs.nav-justified > li > a {
  text-align: center;
  margin-bottom: 5px;
}
.nav-tabs.nav-justified > .dropdown .dropdown-menu {
  top: auto;
  left: auto;
}
@media (min-width: 768px) {
  .nav-tabs.nav-justified > li {
    display: table-cell;
    width: 1%;
  }
  .nav-tabs.nav-justified > li > a {
    margin-bottom: 0;
  }
}
.nav-tabs.nav-justified > li > a {
  margin-right: 0;
  border-radius: 2px;
}
.nav-tabs.nav-justified > .active > a,
.nav-tabs.nav-justified > .active > a:hover,
.nav-tabs.nav-justified > .active > a:focus {
  border: 1px solid #ddd;
}
@media (min-width: 768px) {
  .nav-tabs.nav-justified > li > a {
    border-bottom: 1px solid #ddd;
    border-radius: 2px 2px 0 0;
  }
  .nav-tabs.nav-justified > .active > a,
  .nav-tabs.nav-justified > .active > a:hover,
  .nav-tabs.nav-justified > .active > a:focus {
    border-bottom-color: #fff;
  }
}
.nav-pills > li {
  float: left;
}
.nav-pills > li > a {
  border-radius: 2px;
}
.nav-pills > li + li {
  margin-left: 2px;
}
.nav-pills > li.active > a,
.nav-pills > li.active > a:hover,
.nav-pills > li.active > a:focus {
  color: #fff;
  background-color: #337ab7;
}
.nav-stacked > li {
  float: none;
}
.nav-stacked > li + li {
  margin-top: 2px;
  margin-left: 0;
}
.nav-justified {
  width: 100%;
}
.nav-justified > li {
  float: none;
}
.nav-justified > li > a {
  text-align: center;
  margin-bottom: 5px;
}
.nav-justified > .dropdown .dropdown-menu {
  top: auto;
  left: auto;
}
@media (min-width: 768px) {
  .nav-justified > li {
    display: table-cell;
    width: 1%;
  }
  .nav-justified > li > a {
    margin-bottom: 0;
  }
}
.nav-tabs-justified {
  border-bottom: 0;
}
.nav-tabs-justified > li > a {
  margin-right: 0;
  border-radius: 2px;
}
.nav-tabs-justified > .active > a,
.nav-tabs-justified > .active > a:hover,
.nav-tabs-justified > .active > a:focus {
  border: 1px solid #ddd;
}
@media (min-width: 768px) {
  .nav-tabs-justified > li > a {
    border-bottom: 1px solid #ddd;
    border-radius: 2px 2px 0 0;
  }
  .nav-tabs-justified > .active > a,
  .nav-tabs-justified > .active > a:hover,
  .nav-tabs-justified > .active > a:focus {
    border-bottom-color: #fff;
  }
}
.tab-content > .tab-pane {
  display: none;
}
.tab-content > .active {
  display: block;
}
.nav-tabs .dropdown-menu {
  margin-top: -1px;
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.navbar {
  position: relative;
  min-height: 30px;
  margin-bottom: 18px;
  border: 1px solid transparent;
}
@media (min-width: 541px) {
  .navbar {
    border-radius: 2px;
  }
}
@media (min-width: 541px) {
  .navbar-header {
    float: left;
  }
}
.navbar-collapse {
  overflow-x: visible;
  padding-right: 0px;
  padding-left: 0px;
  border-top: 1px solid transparent;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);
  -webkit-overflow-scrolling: touch;
}
.navbar-collapse.in {
  overflow-y: auto;
}
@media (min-width: 541px) {
  .navbar-collapse {
    width: auto;
    border-top: 0;
    box-shadow: none;
  }
  .navbar-collapse.collapse {
    display: block !important;
    height: auto !important;
    padding-bottom: 0;
    overflow: visible !important;
  }
  .navbar-collapse.in {
    overflow-y: visible;
  }
  .navbar-fixed-top .navbar-collapse,
  .navbar-static-top .navbar-collapse,
  .navbar-fixed-bottom .navbar-collapse {
    padding-left: 0;
    padding-right: 0;
  }
}
.navbar-fixed-top .navbar-collapse,
.navbar-fixed-bottom .navbar-collapse {
  max-height: 340px;
}
@media (max-device-width: 540px) and (orientation: landscape) {
  .navbar-fixed-top .navbar-collapse,
  .navbar-fixed-bottom .navbar-collapse {
    max-height: 200px;
  }
}
.container > .navbar-header,
.container-fluid > .navbar-header,
.container > .navbar-collapse,
.container-fluid > .navbar-collapse {
  margin-right: 0px;
  margin-left: 0px;
}
@media (min-width: 541px) {
  .container > .navbar-header,
  .container-fluid > .navbar-header,
  .container > .navbar-collapse,
  .container-fluid > .navbar-collapse {
    margin-right: 0;
    margin-left: 0;
  }
}
.navbar-static-top {
  z-index: 1000;
  border-width: 0 0 1px;
}
@media (min-width: 541px) {
  .navbar-static-top {
    border-radius: 0;
  }
}
.navbar-fixed-top,
.navbar-fixed-bottom {
  position: fixed;
  right: 0;
  left: 0;
  z-index: 1030;
}
@media (min-width: 541px) {
  .navbar-fixed-top,
  .navbar-fixed-bottom {
    border-radius: 0;
  }
}
.navbar-fixed-top {
  top: 0;
  border-width: 0 0 1px;
}
.navbar-fixed-bottom {
  bottom: 0;
  margin-bottom: 0;
  border-width: 1px 0 0;
}
.navbar-brand {
  float: left;
  padding: 6px 0px;
  font-size: 17px;
  line-height: 18px;
  height: 30px;
}
.navbar-brand:hover,
.navbar-brand:focus {
  text-decoration: none;
}
.navbar-brand > img {
  display: block;
}
@media (min-width: 541px) {
  .navbar > .container .navbar-brand,
  .navbar > .container-fluid .navbar-brand {
    margin-left: 0px;
  }
}
.navbar-toggle {
  position: relative;
  float: right;
  margin-right: 0px;
  padding: 9px 10px;
  margin-top: -2px;
  margin-bottom: -2px;
  background-color: transparent;
  background-image: none;
  border: 1px solid transparent;
  border-radius: 2px;
}
.navbar-toggle:focus {
  outline: 0;
}
.navbar-toggle .icon-bar {
  display: block;
  width: 22px;
  height: 2px;
  border-radius: 1px;
}
.navbar-toggle .icon-bar + .icon-bar {
  margin-top: 4px;
}
@media (min-width: 541px) {
  .navbar-toggle {
    display: none;
  }
}
.navbar-nav {
  margin: 3px 0px;
}
.navbar-nav > li > a {
  padding-top: 10px;
  padding-bottom: 10px;
  line-height: 18px;
}
@media (max-width: 540px) {
  .navbar-nav .open .dropdown-menu {
    position: static;
    float: none;
    width: auto;
    margin-top: 0;
    background-color: transparent;
    border: 0;
    box-shadow: none;
  }
  .navbar-nav .open .dropdown-menu > li > a,
  .navbar-nav .open .dropdown-menu .dropdown-header {
    padding: 5px 15px 5px 25px;
  }
  .navbar-nav .open .dropdown-menu > li > a {
    line-height: 18px;
  }
  .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-nav .open .dropdown-menu > li > a:focus {
    background-image: none;
  }
}
@media (min-width: 541px) {
  .navbar-nav {
    float: left;
    margin: 0;
  }
  .navbar-nav > li {
    float: left;
  }
  .navbar-nav > li > a {
    padding-top: 6px;
    padding-bottom: 6px;
  }
}
.navbar-form {
  margin-left: 0px;
  margin-right: 0px;
  padding: 10px 0px;
  border-top: 1px solid transparent;
  border-bottom: 1px solid transparent;
  -webkit-box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 1px 0 rgba(255, 255, 255, 0.1);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 1px 0 rgba(255, 255, 255, 0.1);
  margin-top: -1px;
  margin-bottom: -1px;
}
@media (min-width: 768px) {
  .navbar-form .form-group {
    display: inline-block;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .form-control {
    display: inline-block;
    width: auto;
    vertical-align: middle;
  }
  .navbar-form .form-control-static {
    display: inline-block;
  }
  .navbar-form .input-group {
    display: inline-table;
    vertical-align: middle;
  }
  .navbar-form .input-group .input-group-addon,
  .navbar-form .input-group .input-group-btn,
  .navbar-form .input-group .form-control {
    width: auto;
  }
  .navbar-form .input-group > .form-control {
    width: 100%;
  }
  .navbar-form .control-label {
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .radio,
  .navbar-form .checkbox {
    display: inline-block;
    margin-top: 0;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .radio label,
  .navbar-form .checkbox label {
    padding-left: 0;
  }
  .navbar-form .radio input[type="radio"],
  .navbar-form .checkbox input[type="checkbox"] {
    position: relative;
    margin-left: 0;
  }
  .navbar-form .has-feedback .form-control-feedback {
    top: 0;
  }
}
@media (max-width: 540px) {
  .navbar-form .form-group {
    margin-bottom: 5px;
  }
  .navbar-form .form-group:last-child {
    margin-bottom: 0;
  }
}
@media (min-width: 541px) {
  .navbar-form {
    width: auto;
    border: 0;
    margin-left: 0;
    margin-right: 0;
    padding-top: 0;
    padding-bottom: 0;
    -webkit-box-shadow: none;
    box-shadow: none;
  }
}
.navbar-nav > li > .dropdown-menu {
  margin-top: 0;
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.navbar-fixed-bottom .navbar-nav > li > .dropdown-menu {
  margin-bottom: 0;
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.navbar-btn {
  margin-top: -1px;
  margin-bottom: -1px;
}
.navbar-btn.btn-sm {
  margin-top: 0px;
  margin-bottom: 0px;
}
.navbar-btn.btn-xs {
  margin-top: 4px;
  margin-bottom: 4px;
}
.navbar-text {
  margin-top: 6px;
  margin-bottom: 6px;
}
@media (min-width: 541px) {
  .navbar-text {
    float: left;
    margin-left: 0px;
    margin-right: 0px;
  }
}
@media (min-width: 541px) {
  .navbar-left {
    float: left !important;
    float: left;
  }
  .navbar-right {
    float: right !important;
    float: right;
    margin-right: 0px;
  }
  .navbar-right ~ .navbar-right {
    margin-right: 0;
  }
}
.navbar-default {
  background-color: #f8f8f8;
  border-color: #e7e7e7;
}
.navbar-default .navbar-brand {
  color: #777;
}
.navbar-default .navbar-brand:hover,
.navbar-default .navbar-brand:focus {
  color: #5e5e5e;
  background-color: transparent;
}
.navbar-default .navbar-text {
  color: #777;
}
.navbar-default .navbar-nav > li > a {
  color: #777;
}
.navbar-default .navbar-nav > li > a:hover,
.navbar-default .navbar-nav > li > a:focus {
  color: #333;
  background-color: transparent;
}
.navbar-default .navbar-nav > .active > a,
.navbar-default .navbar-nav > .active > a:hover,
.navbar-default .navbar-nav > .active > a:focus {
  color: #555;
  background-color: #e7e7e7;
}
.navbar-default .navbar-nav > .disabled > a,
.navbar-default .navbar-nav > .disabled > a:hover,
.navbar-default .navbar-nav > .disabled > a:focus {
  color: #ccc;
  background-color: transparent;
}
.navbar-default .navbar-toggle {
  border-color: #ddd;
}
.navbar-default .navbar-toggle:hover,
.navbar-default .navbar-toggle:focus {
  background-color: #ddd;
}
.navbar-default .navbar-toggle .icon-bar {
  background-color: #888;
}
.navbar-default .navbar-collapse,
.navbar-default .navbar-form {
  border-color: #e7e7e7;
}
.navbar-default .navbar-nav > .open > a,
.navbar-default .navbar-nav > .open > a:hover,
.navbar-default .navbar-nav > .open > a:focus {
  background-color: #e7e7e7;
  color: #555;
}
@media (max-width: 540px) {
  .navbar-default .navbar-nav .open .dropdown-menu > li > a {
    color: #777;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > li > a:focus {
    color: #333;
    background-color: transparent;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a,
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a:focus {
    color: #555;
    background-color: #e7e7e7;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a,
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a:focus {
    color: #ccc;
    background-color: transparent;
  }
}
.navbar-default .navbar-link {
  color: #777;
}
.navbar-default .navbar-link:hover {
  color: #333;
}
.navbar-default .btn-link {
  color: #777;
}
.navbar-default .btn-link:hover,
.navbar-default .btn-link:focus {
  color: #333;
}
.navbar-default .btn-link[disabled]:hover,
fieldset[disabled] .navbar-default .btn-link:hover,
.navbar-default .btn-link[disabled]:focus,
fieldset[disabled] .navbar-default .btn-link:focus {
  color: #ccc;
}
.navbar-inverse {
  background-color: #222;
  border-color: #080808;
}
.navbar-inverse .navbar-brand {
  color: #9d9d9d;
}
.navbar-inverse .navbar-brand:hover,
.navbar-inverse .navbar-brand:focus {
  color: #fff;
  background-color: transparent;
}
.navbar-inverse .navbar-text {
  color: #9d9d9d;
}
.navbar-inverse .navbar-nav > li > a {
  color: #9d9d9d;
}
.navbar-inverse .navbar-nav > li > a:hover,
.navbar-inverse .navbar-nav > li > a:focus {
  color: #fff;
  background-color: transparent;
}
.navbar-inverse .navbar-nav > .active > a,
.navbar-inverse .navbar-nav > .active > a:hover,
.navbar-inverse .navbar-nav > .active > a:focus {
  color: #fff;
  background-color: #080808;
}
.navbar-inverse .navbar-nav > .disabled > a,
.navbar-inverse .navbar-nav > .disabled > a:hover,
.navbar-inverse .navbar-nav > .disabled > a:focus {
  color: #444;
  background-color: transparent;
}
.navbar-inverse .navbar-toggle {
  border-color: #333;
}
.navbar-inverse .navbar-toggle:hover,
.navbar-inverse .navbar-toggle:focus {
  background-color: #333;
}
.navbar-inverse .navbar-toggle .icon-bar {
  background-color: #fff;
}
.navbar-inverse .navbar-collapse,
.navbar-inverse .navbar-form {
  border-color: #101010;
}
.navbar-inverse .navbar-nav > .open > a,
.navbar-inverse .navbar-nav > .open > a:hover,
.navbar-inverse .navbar-nav > .open > a:focus {
  background-color: #080808;
  color: #fff;
}
@media (max-width: 540px) {
  .navbar-inverse .navbar-nav .open .dropdown-menu > .dropdown-header {
    border-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu .divider {
    background-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a {
    color: #9d9d9d;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a:focus {
    color: #fff;
    background-color: transparent;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a:focus {
    color: #fff;
    background-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a:focus {
    color: #444;
    background-color: transparent;
  }
}
.navbar-inverse .navbar-link {
  color: #9d9d9d;
}
.navbar-inverse .navbar-link:hover {
  color: #fff;
}
.navbar-inverse .btn-link {
  color: #9d9d9d;
}
.navbar-inverse .btn-link:hover,
.navbar-inverse .btn-link:focus {
  color: #fff;
}
.navbar-inverse .btn-link[disabled]:hover,
fieldset[disabled] .navbar-inverse .btn-link:hover,
.navbar-inverse .btn-link[disabled]:focus,
fieldset[disabled] .navbar-inverse .btn-link:focus {
  color: #444;
}
.breadcrumb {
  padding: 8px 15px;
  margin-bottom: 18px;
  list-style: none;
  background-color: #f5f5f5;
  border-radius: 2px;
}
.breadcrumb > li {
  display: inline-block;
}
.breadcrumb > li + li:before {
  content: "/\00a0";
  padding: 0 5px;
  color: #5e5e5e;
}
.breadcrumb > .active {
  color: #777777;
}
.pagination {
  display: inline-block;
  padding-left: 0;
  margin: 18px 0;
  border-radius: 2px;
}
.pagination > li {
  display: inline;
}
.pagination > li > a,
.pagination > li > span {
  position: relative;
  float: left;
  padding: 6px 12px;
  line-height: 1.42857143;
  text-decoration: none;
  color: #337ab7;
  background-color: #fff;
  border: 1px solid #ddd;
  margin-left: -1px;
}
.pagination > li:first-child > a,
.pagination > li:first-child > span {
  margin-left: 0;
  border-bottom-left-radius: 2px;
  border-top-left-radius: 2px;
}
.pagination > li:last-child > a,
.pagination > li:last-child > span {
  border-bottom-right-radius: 2px;
  border-top-right-radius: 2px;
}
.pagination > li > a:hover,
.pagination > li > span:hover,
.pagination > li > a:focus,
.pagination > li > span:focus {
  z-index: 2;
  color: #23527c;
  background-color: #eeeeee;
  border-color: #ddd;
}
.pagination > .active > a,
.pagination > .active > span,
.pagination > .active > a:hover,
.pagination > .active > span:hover,
.pagination > .active > a:focus,
.pagination > .active > span:focus {
  z-index: 3;
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
  cursor: default;
}
.pagination > .disabled > span,
.pagination > .disabled > span:hover,
.pagination > .disabled > span:focus,
.pagination > .disabled > a,
.pagination > .disabled > a:hover,
.pagination > .disabled > a:focus {
  color: #777777;
  background-color: #fff;
  border-color: #ddd;
  cursor: not-allowed;
}
.pagination-lg > li > a,
.pagination-lg > li > span {
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
}
.pagination-lg > li:first-child > a,
.pagination-lg > li:first-child > span {
  border-bottom-left-radius: 3px;
  border-top-left-radius: 3px;
}
.pagination-lg > li:last-child > a,
.pagination-lg > li:last-child > span {
  border-bottom-right-radius: 3px;
  border-top-right-radius: 3px;
}
.pagination-sm > li > a,
.pagination-sm > li > span {
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
}
.pagination-sm > li:first-child > a,
.pagination-sm > li:first-child > span {
  border-bottom-left-radius: 1px;
  border-top-left-radius: 1px;
}
.pagination-sm > li:last-child > a,
.pagination-sm > li:last-child > span {
  border-bottom-right-radius: 1px;
  border-top-right-radius: 1px;
}
.pager {
  padding-left: 0;
  margin: 18px 0;
  list-style: none;
  text-align: center;
}
.pager li {
  display: inline;
}
.pager li > a,
.pager li > span {
  display: inline-block;
  padding: 5px 14px;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 15px;
}
.pager li > a:hover,
.pager li > a:focus {
  text-decoration: none;
  background-color: #eeeeee;
}
.pager .next > a,
.pager .next > span {
  float: right;
}
.pager .previous > a,
.pager .previous > span {
  float: left;
}
.pager .disabled > a,
.pager .disabled > a:hover,
.pager .disabled > a:focus,
.pager .disabled > span {
  color: #777777;
  background-color: #fff;
  cursor: not-allowed;
}
.label {
  display: inline;
  padding: .2em .6em .3em;
  font-size: 75%;
  font-weight: bold;
  line-height: 1;
  color: #fff;
  text-align: center;
  white-space: nowrap;
  vertical-align: baseline;
  border-radius: .25em;
}
a.label:hover,
a.label:focus {
  color: #fff;
  text-decoration: none;
  cursor: pointer;
}
.label:empty {
  display: none;
}
.btn .label {
  position: relative;
  top: -1px;
}
.label-default {
  background-color: #777777;
}
.label-default[href]:hover,
.label-default[href]:focus {
  background-color: #5e5e5e;
}
.label-primary {
  background-color: #337ab7;
}
.label-primary[href]:hover,
.label-primary[href]:focus {
  background-color: #286090;
}
.label-success {
  background-color: #5cb85c;
}
.label-success[href]:hover,
.label-success[href]:focus {
  background-color: #449d44;
}
.label-info {
  background-color: #5bc0de;
}
.label-info[href]:hover,
.label-info[href]:focus {
  background-color: #31b0d5;
}
.label-warning {
  background-color: #f0ad4e;
}
.label-warning[href]:hover,
.label-warning[href]:focus {
  background-color: #ec971f;
}
.label-danger {
  background-color: #d9534f;
}
.label-danger[href]:hover,
.label-danger[href]:focus {
  background-color: #c9302c;
}
.badge {
  display: inline-block;
  min-width: 10px;
  padding: 3px 7px;
  font-size: 12px;
  font-weight: bold;
  color: #fff;
  line-height: 1;
  vertical-align: middle;
  white-space: nowrap;
  text-align: center;
  background-color: #777777;
  border-radius: 10px;
}
.badge:empty {
  display: none;
}
.btn .badge {
  position: relative;
  top: -1px;
}
.btn-xs .badge,
.btn-group-xs > .btn .badge {
  top: 0;
  padding: 1px 5px;
}
a.badge:hover,
a.badge:focus {
  color: #fff;
  text-decoration: none;
  cursor: pointer;
}
.list-group-item.active > .badge,
.nav-pills > .active > a > .badge {
  color: #337ab7;
  background-color: #fff;
}
.list-group-item > .badge {
  float: right;
}
.list-group-item > .badge + .badge {
  margin-right: 5px;
}
.nav-pills > li > a > .badge {
  margin-left: 3px;
}
.jumbotron {
  padding-top: 30px;
  padding-bottom: 30px;
  margin-bottom: 30px;
  color: inherit;
  background-color: #eeeeee;
}
.jumbotron h1,
.jumbotron .h1 {
  color: inherit;
}
.jumbotron p {
  margin-bottom: 15px;
  font-size: 20px;
  font-weight: 200;
}
.jumbotron > hr {
  border-top-color: #d5d5d5;
}
.container .jumbotron,
.container-fluid .jumbotron {
  border-radius: 3px;
  padding-left: 0px;
  padding-right: 0px;
}
.jumbotron .container {
  max-width: 100%;
}
@media screen and (min-width: 768px) {
  .jumbotron {
    padding-top: 48px;
    padding-bottom: 48px;
  }
  .container .jumbotron,
  .container-fluid .jumbotron {
    padding-left: 60px;
    padding-right: 60px;
  }
  .jumbotron h1,
  .jumbotron .h1 {
    font-size: 59px;
  }
}
.thumbnail {
  display: block;
  padding: 4px;
  margin-bottom: 18px;
  line-height: 1.42857143;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 2px;
  -webkit-transition: border 0.2s ease-in-out;
  -o-transition: border 0.2s ease-in-out;
  transition: border 0.2s ease-in-out;
}
.thumbnail > img,
.thumbnail a > img {
  margin-left: auto;
  margin-right: auto;
}
a.thumbnail:hover,
a.thumbnail:focus,
a.thumbnail.active {
  border-color: #337ab7;
}
.thumbnail .caption {
  padding: 9px;
  color: #000;
}
.alert {
  padding: 15px;
  margin-bottom: 18px;
  border: 1px solid transparent;
  border-radius: 2px;
}
.alert h4 {
  margin-top: 0;
  color: inherit;
}
.alert .alert-link {
  font-weight: bold;
}
.alert > p,
.alert > ul {
  margin-bottom: 0;
}
.alert > p + p {
  margin-top: 5px;
}
.alert-dismissable,
.alert-dismissible {
  padding-right: 35px;
}
.alert-dismissable .close,
.alert-dismissible .close {
  position: relative;
  top: -2px;
  right: -21px;
  color: inherit;
}
.alert-success {
  background-color: #dff0d8;
  border-color: #d6e9c6;
  color: #3c763d;
}
.alert-success hr {
  border-top-color: #c9e2b3;
}
.alert-success .alert-link {
  color: #2b542c;
}
.alert-info {
  background-color: #d9edf7;
  border-color: #bce8f1;
  color: #31708f;
}
.alert-info hr {
  border-top-color: #a6e1ec;
}
.alert-info .alert-link {
  color: #245269;
}
.alert-warning {
  background-color: #fcf8e3;
  border-color: #faebcc;
  color: #8a6d3b;
}
.alert-warning hr {
  border-top-color: #f7e1b5;
}
.alert-warning .alert-link {
  color: #66512c;
}
.alert-danger {
  background-color: #f2dede;
  border-color: #ebccd1;
  color: #a94442;
}
.alert-danger hr {
  border-top-color: #e4b9c0;
}
.alert-danger .alert-link {
  color: #843534;
}
@-webkit-keyframes progress-bar-stripes {
  from {
    background-position: 40px 0;
  }
  to {
    background-position: 0 0;
  }
}
@keyframes progress-bar-stripes {
  from {
    background-position: 40px 0;
  }
  to {
    background-position: 0 0;
  }
}
.progress {
  overflow: hidden;
  height: 18px;
  margin-bottom: 18px;
  background-color: #f5f5f5;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
  box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
}
.progress-bar {
  float: left;
  width: 0%;
  height: 100%;
  font-size: 12px;
  line-height: 18px;
  color: #fff;
  text-align: center;
  background-color: #337ab7;
  -webkit-box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.15);
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.15);
  -webkit-transition: width 0.6s ease;
  -o-transition: width 0.6s ease;
  transition: width 0.6s ease;
}
.progress-striped .progress-bar,
.progress-bar-striped {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-size: 40px 40px;
}
.progress.active .progress-bar,
.progress-bar.active {
  -webkit-animation: progress-bar-stripes 2s linear infinite;
  -o-animation: progress-bar-stripes 2s linear infinite;
  animation: progress-bar-stripes 2s linear infinite;
}
.progress-bar-success {
  background-color: #5cb85c;
}
.progress-striped .progress-bar-success {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-info {
  background-color: #5bc0de;
}
.progress-striped .progress-bar-info {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-warning {
  background-color: #f0ad4e;
}
.progress-striped .progress-bar-warning {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-danger {
  background-color: #d9534f;
}
.progress-striped .progress-bar-danger {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.media {
  margin-top: 15px;
}
.media:first-child {
  margin-top: 0;
}
.media,
.media-body {
  zoom: 1;
  overflow: hidden;
}
.media-body {
  width: 10000px;
}
.media-object {
  display: block;
}
.media-object.img-thumbnail {
  max-width: none;
}
.media-right,
.media > .pull-right {
  padding-left: 10px;
}
.media-left,
.media > .pull-left {
  padding-right: 10px;
}
.media-left,
.media-right,
.media-body {
  display: table-cell;
  vertical-align: top;
}
.media-middle {
  vertical-align: middle;
}
.media-bottom {
  vertical-align: bottom;
}
.media-heading {
  margin-top: 0;
  margin-bottom: 5px;
}
.media-list {
  padding-left: 0;
  list-style: none;
}
.list-group {
  margin-bottom: 20px;
  padding-left: 0;
}
.list-group-item {
  position: relative;
  display: block;
  padding: 10px 15px;
  margin-bottom: -1px;
  background-color: #fff;
  border: 1px solid #ddd;
}
.list-group-item:first-child {
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
}
.list-group-item:last-child {
  margin-bottom: 0;
  border-bottom-right-radius: 2px;
  border-bottom-left-radius: 2px;
}
a.list-group-item,
button.list-group-item {
  color: #555;
}
a.list-group-item .list-group-item-heading,
button.list-group-item .list-group-item-heading {
  color: #333;
}
a.list-group-item:hover,
button.list-group-item:hover,
a.list-group-item:focus,
button.list-group-item:focus {
  text-decoration: none;
  color: #555;
  background-color: #f5f5f5;
}
button.list-group-item {
  width: 100%;
  text-align: left;
}
.list-group-item.disabled,
.list-group-item.disabled:hover,
.list-group-item.disabled:focus {
  background-color: #eeeeee;
  color: #777777;
  cursor: not-allowed;
}
.list-group-item.disabled .list-group-item-heading,
.list-group-item.disabled:hover .list-group-item-heading,
.list-group-item.disabled:focus .list-group-item-heading {
  color: inherit;
}
.list-group-item.disabled .list-group-item-text,
.list-group-item.disabled:hover .list-group-item-text,
.list-group-item.disabled:focus .list-group-item-text {
  color: #777777;
}
.list-group-item.active,
.list-group-item.active:hover,
.list-group-item.active:focus {
  z-index: 2;
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
}
.list-group-item.active .list-group-item-heading,
.list-group-item.active:hover .list-group-item-heading,
.list-group-item.active:focus .list-group-item-heading,
.list-group-item.active .list-group-item-heading > small,
.list-group-item.active:hover .list-group-item-heading > small,
.list-group-item.active:focus .list-group-item-heading > small,
.list-group-item.active .list-group-item-heading > .small,
.list-group-item.active:hover .list-group-item-heading > .small,
.list-group-item.active:focus .list-group-item-heading > .small {
  color: inherit;
}
.list-group-item.active .list-group-item-text,
.list-group-item.active:hover .list-group-item-text,
.list-group-item.active:focus .list-group-item-text {
  color: #c7ddef;
}
.list-group-item-success {
  color: #3c763d;
  background-color: #dff0d8;
}
a.list-group-item-success,
button.list-group-item-success {
  color: #3c763d;
}
a.list-group-item-success .list-group-item-heading,
button.list-group-item-success .list-group-item-heading {
  color: inherit;
}
a.list-group-item-success:hover,
button.list-group-item-success:hover,
a.list-group-item-success:focus,
button.list-group-item-success:focus {
  color: #3c763d;
  background-color: #d0e9c6;
}
a.list-group-item-success.active,
button.list-group-item-success.active,
a.list-group-item-success.active:hover,
button.list-group-item-success.active:hover,
a.list-group-item-success.active:focus,
button.list-group-item-success.active:focus {
  color: #fff;
  background-color: #3c763d;
  border-color: #3c763d;
}
.list-group-item-info {
  color: #31708f;
  background-color: #d9edf7;
}
a.list-group-item-info,
button.list-group-item-info {
  color: #31708f;
}
a.list-group-item-info .list-group-item-heading,
button.list-group-item-info .list-group-item-heading {
  color: inherit;
}
a.list-group-item-info:hover,
button.list-group-item-info:hover,
a.list-group-item-info:focus,
button.list-group-item-info:focus {
  color: #31708f;
  background-color: #c4e3f3;
}
a.list-group-item-info.active,
button.list-group-item-info.active,
a.list-group-item-info.active:hover,
button.list-group-item-info.active:hover,
a.list-group-item-info.active:focus,
button.list-group-item-info.active:focus {
  color: #fff;
  background-color: #31708f;
  border-color: #31708f;
}
.list-group-item-warning {
  color: #8a6d3b;
  background-color: #fcf8e3;
}
a.list-group-item-warning,
button.list-group-item-warning {
  color: #8a6d3b;
}
a.list-group-item-warning .list-group-item-heading,
button.list-group-item-warning .list-group-item-heading {
  color: inherit;
}
a.list-group-item-warning:hover,
button.list-group-item-warning:hover,
a.list-group-item-warning:focus,
button.list-group-item-warning:focus {
  color: #8a6d3b;
  background-color: #faf2cc;
}
a.list-group-item-warning.active,
button.list-group-item-warning.active,
a.list-group-item-warning.active:hover,
button.list-group-item-warning.active:hover,
a.list-group-item-warning.active:focus,
button.list-group-item-warning.active:focus {
  color: #fff;
  background-color: #8a6d3b;
  border-color: #8a6d3b;
}
.list-group-item-danger {
  color: #a94442;
  background-color: #f2dede;
}
a.list-group-item-danger,
button.list-group-item-danger {
  color: #a94442;
}
a.list-group-item-danger .list-group-item-heading,
button.list-group-item-danger .list-group-item-heading {
  color: inherit;
}
a.list-group-item-danger:hover,
button.list-group-item-danger:hover,
a.list-group-item-danger:focus,
button.list-group-item-danger:focus {
  color: #a94442;
  background-color: #ebcccc;
}
a.list-group-item-danger.active,
button.list-group-item-danger.active,
a.list-group-item-danger.active:hover,
button.list-group-item-danger.active:hover,
a.list-group-item-danger.active:focus,
button.list-group-item-danger.active:focus {
  color: #fff;
  background-color: #a94442;
  border-color: #a94442;
}
.list-group-item-heading {
  margin-top: 0;
  margin-bottom: 5px;
}
.list-group-item-text {
  margin-bottom: 0;
  line-height: 1.3;
}
.panel {
  margin-bottom: 18px;
  background-color: #fff;
  border: 1px solid transparent;
  border-radius: 2px;
  -webkit-box-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);
  box-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);
}
.panel-body {
  padding: 15px;
}
.panel-heading {
  padding: 10px 15px;
  border-bottom: 1px solid transparent;
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel-heading > .dropdown .dropdown-toggle {
  color: inherit;
}
.panel-title {
  margin-top: 0;
  margin-bottom: 0;
  font-size: 15px;
  color: inherit;
}
.panel-title > a,
.panel-title > small,
.panel-title > .small,
.panel-title > small > a,
.panel-title > .small > a {
  color: inherit;
}
.panel-footer {
  padding: 10px 15px;
  background-color: #f5f5f5;
  border-top: 1px solid #ddd;
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .list-group,
.panel > .panel-collapse > .list-group {
  margin-bottom: 0;
}
.panel > .list-group .list-group-item,
.panel > .panel-collapse > .list-group .list-group-item {
  border-width: 1px 0;
  border-radius: 0;
}
.panel > .list-group:first-child .list-group-item:first-child,
.panel > .panel-collapse > .list-group:first-child .list-group-item:first-child {
  border-top: 0;
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel > .list-group:last-child .list-group-item:last-child,
.panel > .panel-collapse > .list-group:last-child .list-group-item:last-child {
  border-bottom: 0;
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .panel-heading + .panel-collapse > .list-group .list-group-item:first-child {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.panel-heading + .list-group .list-group-item:first-child {
  border-top-width: 0;
}
.list-group + .panel-footer {
  border-top-width: 0;
}
.panel > .table,
.panel > .table-responsive > .table,
.panel > .panel-collapse > .table {
  margin-bottom: 0;
}
.panel > .table caption,
.panel > .table-responsive > .table caption,
.panel > .panel-collapse > .table caption {
  padding-left: 15px;
  padding-right: 15px;
}
.panel > .table:first-child,
.panel > .table-responsive:first-child > .table:first-child {
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child {
  border-top-left-radius: 1px;
  border-top-right-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child td:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child td:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child td:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child td:first-child,
.panel > .table:first-child > thead:first-child > tr:first-child th:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child th:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child th:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child th:first-child {
  border-top-left-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child td:last-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child td:last-child,
.panel > .table:first-child > tbody:first-child > tr:first-child td:last-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child td:last-child,
.panel > .table:first-child > thead:first-child > tr:first-child th:last-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child th:last-child,
.panel > .table:first-child > tbody:first-child > tr:first-child th:last-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child th:last-child {
  border-top-right-radius: 1px;
}
.panel > .table:last-child,
.panel > .table-responsive:last-child > .table:last-child {
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child {
  border-bottom-left-radius: 1px;
  border-bottom-right-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child td:first-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child td:first-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child td:first-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child td:first-child,
.panel > .table:last-child > tbody:last-child > tr:last-child th:first-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child th:first-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child th:first-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child th:first-child {
  border-bottom-left-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child td:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child td:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child td:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child td:last-child,
.panel > .table:last-child > tbody:last-child > tr:last-child th:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child th:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child th:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child th:last-child {
  border-bottom-right-radius: 1px;
}
.panel > .panel-body + .table,
.panel > .panel-body + .table-responsive,
.panel > .table + .panel-body,
.panel > .table-responsive + .panel-body {
  border-top: 1px solid #ddd;
}
.panel > .table > tbody:first-child > tr:first-child th,
.panel > .table > tbody:first-child > tr:first-child td {
  border-top: 0;
}
.panel > .table-bordered,
.panel > .table-responsive > .table-bordered {
  border: 0;
}
.panel > .table-bordered > thead > tr > th:first-child,
.panel > .table-responsive > .table-bordered > thead > tr > th:first-child,
.panel > .table-bordered > tbody > tr > th:first-child,
.panel > .table-responsive > .table-bordered > tbody > tr > th:first-child,
.panel > .table-bordered > tfoot > tr > th:first-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > th:first-child,
.panel > .table-bordered > thead > tr > td:first-child,
.panel > .table-responsive > .table-bordered > thead > tr > td:first-child,
.panel > .table-bordered > tbody > tr > td:first-child,
.panel > .table-responsive > .table-bordered > tbody > tr > td:first-child,
.panel > .table-bordered > tfoot > tr > td:first-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > td:first-child {
  border-left: 0;
}
.panel > .table-bordered > thead > tr > th:last-child,
.panel > .table-responsive > .table-bordered > thead > tr > th:last-child,
.panel > .table-bordered > tbody > tr > th:last-child,
.panel > .table-responsive > .table-bordered > tbody > tr > th:last-child,
.panel > .table-bordered > tfoot > tr > th:last-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > th:last-child,
.panel > .table-bordered > thead > tr > td:last-child,
.panel > .table-responsive > .table-bordered > thead > tr > td:last-child,
.panel > .table-bordered > tbody > tr > td:last-child,
.panel > .table-responsive > .table-bordered > tbody > tr > td:last-child,
.panel > .table-bordered > tfoot > tr > td:last-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > td:last-child {
  border-right: 0;
}
.panel > .table-bordered > thead > tr:first-child > td,
.panel > .table-responsive > .table-bordered > thead > tr:first-child > td,
.panel > .table-bordered > tbody > tr:first-child > td,
.panel > .table-responsive > .table-bordered > tbody > tr:first-child > td,
.panel > .table-bordered > thead > tr:first-child > th,
.panel > .table-responsive > .table-bordered > thead > tr:first-child > th,
.panel > .table-bordered > tbody > tr:first-child > th,
.panel > .table-responsive > .table-bordered > tbody > tr:first-child > th {
  border-bottom: 0;
}
.panel > .table-bordered > tbody > tr:last-child > td,
.panel > .table-responsive > .table-bordered > tbody > tr:last-child > td,
.panel > .table-bordered > tfoot > tr:last-child > td,
.panel > .table-responsive > .table-bordered > tfoot > tr:last-child > td,
.panel > .table-bordered > tbody > tr:last-child > th,
.panel > .table-responsive > .table-bordered > tbody > tr:last-child > th,
.panel > .table-bordered > tfoot > tr:last-child > th,
.panel > .table-responsive > .table-bordered > tfoot > tr:last-child > th {
  border-bottom: 0;
}
.panel > .table-responsive {
  border: 0;
  margin-bottom: 0;
}
.panel-group {
  margin-bottom: 18px;
}
.panel-group .panel {
  margin-bottom: 0;
  border-radius: 2px;
}
.panel-group .panel + .panel {
  margin-top: 5px;
}
.panel-group .panel-heading {
  border-bottom: 0;
}
.panel-group .panel-heading + .panel-collapse > .panel-body,
.panel-group .panel-heading + .panel-collapse > .list-group {
  border-top: 1px solid #ddd;
}
.panel-group .panel-footer {
  border-top: 0;
}
.panel-group .panel-footer + .panel-collapse .panel-body {
  border-bottom: 1px solid #ddd;
}
.panel-default {
  border-color: #ddd;
}
.panel-default > .panel-heading {
  color: #333333;
  background-color: #f5f5f5;
  border-color: #ddd;
}
.panel-default > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #ddd;
}
.panel-default > .panel-heading .badge {
  color: #f5f5f5;
  background-color: #333333;
}
.panel-default > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #ddd;
}
.panel-primary {
  border-color: #337ab7;
}
.panel-primary > .panel-heading {
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
}
.panel-primary > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #337ab7;
}
.panel-primary > .panel-heading .badge {
  color: #337ab7;
  background-color: #fff;
}
.panel-primary > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #337ab7;
}
.panel-success {
  border-color: #d6e9c6;
}
.panel-success > .panel-heading {
  color: #3c763d;
  background-color: #dff0d8;
  border-color: #d6e9c6;
}
.panel-success > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #d6e9c6;
}
.panel-success > .panel-heading .badge {
  color: #dff0d8;
  background-color: #3c763d;
}
.panel-success > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #d6e9c6;
}
.panel-info {
  border-color: #bce8f1;
}
.panel-info > .panel-heading {
  color: #31708f;
  background-color: #d9edf7;
  border-color: #bce8f1;
}
.panel-info > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #bce8f1;
}
.panel-info > .panel-heading .badge {
  color: #d9edf7;
  background-color: #31708f;
}
.panel-info > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #bce8f1;
}
.panel-warning {
  border-color: #faebcc;
}
.panel-warning > .panel-heading {
  color: #8a6d3b;
  background-color: #fcf8e3;
  border-color: #faebcc;
}
.panel-warning > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #faebcc;
}
.panel-warning > .panel-heading .badge {
  color: #fcf8e3;
  background-color: #8a6d3b;
}
.panel-warning > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #faebcc;
}
.panel-danger {
  border-color: #ebccd1;
}
.panel-danger > .panel-heading {
  color: #a94442;
  background-color: #f2dede;
  border-color: #ebccd1;
}
.panel-danger > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #ebccd1;
}
.panel-danger > .panel-heading .badge {
  color: #f2dede;
  background-color: #a94442;
}
.panel-danger > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #ebccd1;
}
.embed-responsive {
  position: relative;
  display: block;
  height: 0;
  padding: 0;
  overflow: hidden;
}
.embed-responsive .embed-responsive-item,
.embed-responsive iframe,
.embed-responsive embed,
.embed-responsive object,
.embed-responsive video {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  height: 100%;
  width: 100%;
  border: 0;
}
.embed-responsive-16by9 {
  padding-bottom: 56.25%;
}
.embed-responsive-4by3 {
  padding-bottom: 75%;
}
.well {
  min-height: 20px;
  padding: 19px;
  margin-bottom: 20px;
  background-color: #f5f5f5;
  border: 1px solid #e3e3e3;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.05);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.05);
}
.well blockquote {
  border-color: #ddd;
  border-color: rgba(0, 0, 0, 0.15);
}
.well-lg {
  padding: 24px;
  border-radius: 3px;
}
.well-sm {
  padding: 9px;
  border-radius: 1px;
}
.close {
  float: right;
  font-size: 19.5px;
  font-weight: bold;
  line-height: 1;
  color: #000;
  text-shadow: 0 1px 0 #fff;
  opacity: 0.2;
  filter: alpha(opacity=20);
}
.close:hover,
.close:focus {
  color: #000;
  text-decoration: none;
  cursor: pointer;
  opacity: 0.5;
  filter: alpha(opacity=50);
}
button.close {
  padding: 0;
  cursor: pointer;
  background: transparent;
  border: 0;
  -webkit-appearance: none;
}
.modal-open {
  overflow: hidden;
}
.modal {
  display: none;
  overflow: hidden;
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 1050;
  -webkit-overflow-scrolling: touch;
  outline: 0;
}
.modal.fade .modal-dialog {
  -webkit-transform: translate(0, -25%);
  -ms-transform: translate(0, -25%);
  -o-transform: translate(0, -25%);
  transform: translate(0, -25%);
  -webkit-transition: -webkit-transform 0.3s ease-out;
  -moz-transition: -moz-transform 0.3s ease-out;
  -o-transition: -o-transform 0.3s ease-out;
  transition: transform 0.3s ease-out;
}
.modal.in .modal-dialog {
  -webkit-transform: translate(0, 0);
  -ms-transform: translate(0, 0);
  -o-transform: translate(0, 0);
  transform: translate(0, 0);
}
.modal-open .modal {
  overflow-x: hidden;
  overflow-y: auto;
}
.modal-dialog {
  position: relative;
  width: auto;
  margin: 10px;
}
.modal-content {
  position: relative;
  background-color: #fff;
  border: 1px solid #999;
  border: 1px solid rgba(0, 0, 0, 0.2);
  border-radius: 3px;
  -webkit-box-shadow: 0 3px 9px rgba(0, 0, 0, 0.5);
  box-shadow: 0 3px 9px rgba(0, 0, 0, 0.5);
  background-clip: padding-box;
  outline: 0;
}
.modal-backdrop {
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 1040;
  background-color: #000;
}
.modal-backdrop.fade {
  opacity: 0;
  filter: alpha(opacity=0);
}
.modal-backdrop.in {
  opacity: 0.5;
  filter: alpha(opacity=50);
}
.modal-header {
  padding: 15px;
  border-bottom: 1px solid #e5e5e5;
}
.modal-header .close {
  margin-top: -2px;
}
.modal-title {
  margin: 0;
  line-height: 1.42857143;
}
.modal-body {
  position: relative;
  padding: 15px;
}
.modal-footer {
  padding: 15px;
  text-align: right;
  border-top: 1px solid #e5e5e5;
}
.modal-footer .btn + .btn {
  margin-left: 5px;
  margin-bottom: 0;
}
.modal-footer .btn-group .btn + .btn {
  margin-left: -1px;
}
.modal-footer .btn-block + .btn-block {
  margin-left: 0;
}
.modal-scrollbar-measure {
  position: absolute;
  top: -9999px;
  width: 50px;
  height: 50px;
  overflow: scroll;
}
@media (min-width: 768px) {
  .modal-dialog {
    width: 600px;
    margin: 30px auto;
  }
  .modal-content {
    -webkit-box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
  }
  .modal-sm {
    width: 300px;
  }
}
@media (min-width: 992px) {
  .modal-lg {
    width: 900px;
  }
}
.tooltip {
  position: absolute;
  z-index: 1070;
  display: block;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-style: normal;
  font-weight: normal;
  letter-spacing: normal;
  line-break: auto;
  line-height: 1.42857143;
  text-align: left;
  text-align: start;
  text-decoration: none;
  text-shadow: none;
  text-transform: none;
  white-space: normal;
  word-break: normal;
  word-spacing: normal;
  word-wrap: normal;
  font-size: 12px;
  opacity: 0;
  filter: alpha(opacity=0);
}
.tooltip.in {
  opacity: 0.9;
  filter: alpha(opacity=90);
}
.tooltip.top {
  margin-top: -3px;
  padding: 5px 0;
}
.tooltip.right {
  margin-left: 3px;
  padding: 0 5px;
}
.tooltip.bottom {
  margin-top: 3px;
  padding: 5px 0;
}
.tooltip.left {
  margin-left: -3px;
  padding: 0 5px;
}
.tooltip-inner {
  max-width: 200px;
  padding: 3px 8px;
  color: #fff;
  text-align: center;
  background-color: #000;
  border-radius: 2px;
}
.tooltip-arrow {
  position: absolute;
  width: 0;
  height: 0;
  border-color: transparent;
  border-style: solid;
}
.tooltip.top .tooltip-arrow {
  bottom: 0;
  left: 50%;
  margin-left: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.top-left .tooltip-arrow {
  bottom: 0;
  right: 5px;
  margin-bottom: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.top-right .tooltip-arrow {
  bottom: 0;
  left: 5px;
  margin-bottom: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.right .tooltip-arrow {
  top: 50%;
  left: 0;
  margin-top: -5px;
  border-width: 5px 5px 5px 0;
  border-right-color: #000;
}
.tooltip.left .tooltip-arrow {
  top: 50%;
  right: 0;
  margin-top: -5px;
  border-width: 5px 0 5px 5px;
  border-left-color: #000;
}
.tooltip.bottom .tooltip-arrow {
  top: 0;
  left: 50%;
  margin-left: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.tooltip.bottom-left .tooltip-arrow {
  top: 0;
  right: 5px;
  margin-top: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.tooltip.bottom-right .tooltip-arrow {
  top: 0;
  left: 5px;
  margin-top: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.popover {
  position: absolute;
  top: 0;
  left: 0;
  z-index: 1060;
  display: none;
  max-width: 276px;
  padding: 1px;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-style: normal;
  font-weight: normal;
  letter-spacing: normal;
  line-break: auto;
  line-height: 1.42857143;
  text-align: left;
  text-align: start;
  text-decoration: none;
  text-shadow: none;
  text-transform: none;
  white-space: normal;
  word-break: normal;
  word-spacing: normal;
  word-wrap: normal;
  font-size: 13px;
  background-color: #fff;
  background-clip: padding-box;
  border: 1px solid #ccc;
  border: 1px solid rgba(0, 0, 0, 0.2);
  border-radius: 3px;
  -webkit-box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
  box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
}
.popover.top {
  margin-top: -10px;
}
.popover.right {
  margin-left: 10px;
}
.popover.bottom {
  margin-top: 10px;
}
.popover.left {
  margin-left: -10px;
}
.popover-title {
  margin: 0;
  padding: 8px 14px;
  font-size: 13px;
  background-color: #f7f7f7;
  border-bottom: 1px solid #ebebeb;
  border-radius: 2px 2px 0 0;
}
.popover-content {
  padding: 9px 14px;
}
.popover > .arrow,
.popover > .arrow:after {
  position: absolute;
  display: block;
  width: 0;
  height: 0;
  border-color: transparent;
  border-style: solid;
}
.popover > .arrow {
  border-width: 11px;
}
.popover > .arrow:after {
  border-width: 10px;
  content: "";
}
.popover.top > .arrow {
  left: 50%;
  margin-left: -11px;
  border-bottom-width: 0;
  border-top-color: #999999;
  border-top-color: rgba(0, 0, 0, 0.25);
  bottom: -11px;
}
.popover.top > .arrow:after {
  content: " ";
  bottom: 1px;
  margin-left: -10px;
  border-bottom-width: 0;
  border-top-color: #fff;
}
.popover.right > .arrow {
  top: 50%;
  left: -11px;
  margin-top: -11px;
  border-left-width: 0;
  border-right-color: #999999;
  border-right-color: rgba(0, 0, 0, 0.25);
}
.popover.right > .arrow:after {
  content: " ";
  left: 1px;
  bottom: -10px;
  border-left-width: 0;
  border-right-color: #fff;
}
.popover.bottom > .arrow {
  left: 50%;
  margin-left: -11px;
  border-top-width: 0;
  border-bottom-color: #999999;
  border-bottom-color: rgba(0, 0, 0, 0.25);
  top: -11px;
}
.popover.bottom > .arrow:after {
  content: " ";
  top: 1px;
  margin-left: -10px;
  border-top-width: 0;
  border-bottom-color: #fff;
}
.popover.left > .arrow {
  top: 50%;
  right: -11px;
  margin-top: -11px;
  border-right-width: 0;
  border-left-color: #999999;
  border-left-color: rgba(0, 0, 0, 0.25);
}
.popover.left > .arrow:after {
  content: " ";
  right: 1px;
  border-right-width: 0;
  border-left-color: #fff;
  bottom: -10px;
}
.carousel {
  position: relative;
}
.carousel-inner {
  position: relative;
  overflow: hidden;
  width: 100%;
}
.carousel-inner > .item {
  display: none;
  position: relative;
  -webkit-transition: 0.6s ease-in-out left;
  -o-transition: 0.6s ease-in-out left;
  transition: 0.6s ease-in-out left;
}
.carousel-inner > .item > img,
.carousel-inner > .item > a > img {
  line-height: 1;
}
@media all and (transform-3d), (-webkit-transform-3d) {
  .carousel-inner > .item {
    -webkit-transition: -webkit-transform 0.6s ease-in-out;
    -moz-transition: -moz-transform 0.6s ease-in-out;
    -o-transition: -o-transform 0.6s ease-in-out;
    transition: transform 0.6s ease-in-out;
    -webkit-backface-visibility: hidden;
    -moz-backface-visibility: hidden;
    backface-visibility: hidden;
    -webkit-perspective: 1000px;
    -moz-perspective: 1000px;
    perspective: 1000px;
  }
  .carousel-inner > .item.next,
  .carousel-inner > .item.active.right {
    -webkit-transform: translate3d(100%, 0, 0);
    transform: translate3d(100%, 0, 0);
    left: 0;
  }
  .carousel-inner > .item.prev,
  .carousel-inner > .item.active.left {
    -webkit-transform: translate3d(-100%, 0, 0);
    transform: translate3d(-100%, 0, 0);
    left: 0;
  }
  .carousel-inner > .item.next.left,
  .carousel-inner > .item.prev.right,
  .carousel-inner > .item.active {
    -webkit-transform: translate3d(0, 0, 0);
    transform: translate3d(0, 0, 0);
    left: 0;
  }
}
.carousel-inner > .active,
.carousel-inner > .next,
.carousel-inner > .prev {
  display: block;
}
.carousel-inner > .active {
  left: 0;
}
.carousel-inner > .next,
.carousel-inner > .prev {
  position: absolute;
  top: 0;
  width: 100%;
}
.carousel-inner > .next {
  left: 100%;
}
.carousel-inner > .prev {
  left: -100%;
}
.carousel-inner > .next.left,
.carousel-inner > .prev.right {
  left: 0;
}
.carousel-inner > .active.left {
  left: -100%;
}
.carousel-inner > .active.right {
  left: 100%;
}
.carousel-control {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  width: 15%;
  opacity: 0.5;
  filter: alpha(opacity=50);
  font-size: 20px;
  color: #fff;
  text-align: center;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
  background-color: rgba(0, 0, 0, 0);
}
.carousel-control.left {
  background-image: -webkit-linear-gradient(left, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-image: -o-linear-gradient(left, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-image: linear-gradient(to right, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-repeat: repeat-x;
  filter: progid:DXImageTransform.Microsoft.gradient(startColorstr='#80000000', endColorstr='#00000000', GradientType=1);
}
.carousel-control.right {
  left: auto;
  right: 0;
  background-image: -webkit-linear-gradient(left, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-image: -o-linear-gradient(left, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-image: linear-gradient(to right, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-repeat: repeat-x;
  filter: progid:DXImageTransform.Microsoft.gradient(startColorstr='#00000000', endColorstr='#80000000', GradientType=1);
}
.carousel-control:hover,
.carousel-control:focus {
  outline: 0;
  color: #fff;
  text-decoration: none;
  opacity: 0.9;
  filter: alpha(opacity=90);
}
.carousel-control .icon-prev,
.carousel-control .icon-next,
.carousel-control .glyphicon-chevron-left,
.carousel-control .glyphicon-chevron-right {
  position: absolute;
  top: 50%;
  margin-top: -10px;
  z-index: 5;
  display: inline-block;
}
.carousel-control .icon-prev,
.carousel-control .glyphicon-chevron-left {
  left: 50%;
  margin-left: -10px;
}
.carousel-control .icon-next,
.carousel-control .glyphicon-chevron-right {
  right: 50%;
  margin-right: -10px;
}
.carousel-control .icon-prev,
.carousel-control .icon-next {
  width: 20px;
  height: 20px;
  line-height: 1;
  font-family: serif;
}
.carousel-control .icon-prev:before {
  content: '\2039';
}
.carousel-control .icon-next:before {
  content: '\203a';
}
.carousel-indicators {
  position: absolute;
  bottom: 10px;
  left: 50%;
  z-index: 15;
  width: 60%;
  margin-left: -30%;
  padding-left: 0;
  list-style: none;
  text-align: center;
}
.carousel-indicators li {
  display: inline-block;
  width: 10px;
  height: 10px;
  margin: 1px;
  text-indent: -999px;
  border: 1px solid #fff;
  border-radius: 10px;
  cursor: pointer;
  background-color: #000 \9;
  background-color: rgba(0, 0, 0, 0);
}
.carousel-indicators .active {
  margin: 0;
  width: 12px;
  height: 12px;
  background-color: #fff;
}
.carousel-caption {
  position: absolute;
  left: 15%;
  right: 15%;
  bottom: 20px;
  z-index: 10;
  padding-top: 20px;
  padding-bottom: 20px;
  color: #fff;
  text-align: center;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
}
.carousel-caption .btn {
  text-shadow: none;
}
@media screen and (min-width: 768px) {
  .carousel-control .glyphicon-chevron-left,
  .carousel-control .glyphicon-chevron-right,
  .carousel-control .icon-prev,
  .carousel-control .icon-next {
    width: 30px;
    height: 30px;
    margin-top: -10px;
    font-size: 30px;
  }
  .carousel-control .glyphicon-chevron-left,
  .carousel-control .icon-prev {
    margin-left: -10px;
  }
  .carousel-control .glyphicon-chevron-right,
  .carousel-control .icon-next {
    margin-right: -10px;
  }
  .carousel-caption {
    left: 20%;
    right: 20%;
    padding-bottom: 30px;
  }
  .carousel-indicators {
    bottom: 20px;
  }
}
.clearfix:before,
.clearfix:after,
.dl-horizontal dd:before,
.dl-horizontal dd:after,
.container:before,
.container:after,
.container-fluid:before,
.container-fluid:after,
.row:before,
.row:after,
.form-horizontal .form-group:before,
.form-horizontal .form-group:after,
.btn-toolbar:before,
.btn-toolbar:after,
.btn-group-vertical > .btn-group:before,
.btn-group-vertical > .btn-group:after,
.nav:before,
.nav:after,
.navbar:before,
.navbar:after,
.navbar-header:before,
.navbar-header:after,
.navbar-collapse:before,
.navbar-collapse:after,
.pager:before,
.pager:after,
.panel-body:before,
.panel-body:after,
.modal-header:before,
.modal-header:after,
.modal-footer:before,
.modal-footer:after,
.item_buttons:before,
.item_buttons:after {
  content: " ";
  display: table;
}
.clearfix:after,
.dl-horizontal dd:after,
.container:after,
.container-fluid:after,
.row:after,
.form-horizontal .form-group:after,
.btn-toolbar:after,
.btn-group-vertical > .btn-group:after,
.nav:after,
.navbar:after,
.navbar-header:after,
.navbar-collapse:after,
.pager:after,
.panel-body:after,
.modal-header:after,
.modal-footer:after,
.item_buttons:after {
  clear: both;
}
.center-block {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
.pull-right {
  float: right !important;
}
.pull-left {
  float: left !important;
}
.hide {
  display: none !important;
}
.show {
  display: block !important;
}
.invisible {
  visibility: hidden;
}
.text-hide {
  font: 0/0 a;
  color: transparent;
  text-shadow: none;
  background-color: transparent;
  border: 0;
}
.hidden {
  display: none !important;
}
.affix {
  position: fixed;
}
@-ms-viewport {
  width: device-width;
}
.visible-xs,
.visible-sm,
.visible-md,
.visible-lg {
  display: none !important;
}
.visible-xs-block,
.visible-xs-inline,
.visible-xs-inline-block,
.visible-sm-block,
.visible-sm-inline,
.visible-sm-inline-block,
.visible-md-block,
.visible-md-inline,
.visible-md-inline-block,
.visible-lg-block,
.visible-lg-inline,
.visible-lg-inline-block {
  display: none !important;
}
@media (max-width: 767px) {
  .visible-xs {
    display: block !important;
  }
  table.visible-xs {
    display: table !important;
  }
  tr.visible-xs {
    display: table-row !important;
  }
  th.visible-xs,
  td.visible-xs {
    display: table-cell !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-block {
    display: block !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-inline {
    display: inline !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm {
    display: block !important;
  }
  table.visible-sm {
    display: table !important;
  }
  tr.visible-sm {
    display: table-row !important;
  }
  th.visible-sm,
  td.visible-sm {
    display: table-cell !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-block {
    display: block !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-inline {
    display: inline !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md {
    display: block !important;
  }
  table.visible-md {
    display: table !important;
  }
  tr.visible-md {
    display: table-row !important;
  }
  th.visible-md,
  td.visible-md {
    display: table-cell !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-block {
    display: block !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-inline {
    display: inline !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg {
    display: block !important;
  }
  table.visible-lg {
    display: table !important;
  }
  tr.visible-lg {
    display: table-row !important;
  }
  th.visible-lg,
  td.visible-lg {
    display: table-cell !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-block {
    display: block !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-inline {
    display: inline !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-inline-block {
    display: inline-block !important;
  }
}
@media (max-width: 767px) {
  .hidden-xs {
    display: none !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .hidden-sm {
    display: none !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .hidden-md {
    display: none !important;
  }
}
@media (min-width: 1200px) {
  .hidden-lg {
    display: none !important;
  }
}
.visible-print {
  display: none !important;
}
@media print {
  .visible-print {
    display: block !important;
  }
  table.visible-print {
    display: table !important;
  }
  tr.visible-print {
    display: table-row !important;
  }
  th.visible-print,
  td.visible-print {
    display: table-cell !important;
  }
}
.visible-print-block {
  display: none !important;
}
@media print {
  .visible-print-block {
    display: block !important;
  }
}
.visible-print-inline {
  display: none !important;
}
@media print {
  .visible-print-inline {
    display: inline !important;
  }
}
.visible-print-inline-block {
  display: none !important;
}
@media print {
  .visible-print-inline-block {
    display: inline-block !important;
  }
}
@media print {
  .hidden-print {
    display: none !important;
  }
}
/*!
*
* Font Awesome
*
*/
/*!
 *  Font Awesome 4.7.0 by @davegandy - http://fontawesome.io - @fontawesome
 *  License - http://fontawesome.io/license (Font: SIL OFL 1.1, CSS: MIT License)
 */
/* FONT PATH
 * -------------------------- */
@font-face {
  font-family: 'FontAwesome';
  src: url('../components/font-awesome/fonts/fontawesome-webfont.eot?v=4.7.0');
  src: url('../components/font-awesome/fonts/fontawesome-webfont.eot?#iefix&v=4.7.0') format('embedded-opentype'), url('../components/font-awesome/fonts/fontawesome-webfont.woff2?v=4.7.0') format('woff2'), url('../components/font-awesome/fonts/fontawesome-webfont.woff?v=4.7.0') format('woff'), url('../components/font-awesome/fonts/fontawesome-webfont.ttf?v=4.7.0') format('truetype'), url('../components/font-awesome/fonts/fontawesome-webfont.svg?v=4.7.0#fontawesomeregular') format('svg');
  font-weight: normal;
  font-style: normal;
}
.fa {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
/* makes the font 33% larger relative to the icon container */
.fa-lg {
  font-size: 1.33333333em;
  line-height: 0.75em;
  vertical-align: -15%;
}
.fa-2x {
  font-size: 2em;
}
.fa-3x {
  font-size: 3em;
}
.fa-4x {
  font-size: 4em;
}
.fa-5x {
  font-size: 5em;
}
.fa-fw {
  width: 1.28571429em;
  text-align: center;
}
.fa-ul {
  padding-left: 0;
  margin-left: 2.14285714em;
  list-style-type: none;
}
.fa-ul > li {
  position: relative;
}
.fa-li {
  position: absolute;
  left: -2.14285714em;
  width: 2.14285714em;
  top: 0.14285714em;
  text-align: center;
}
.fa-li.fa-lg {
  left: -1.85714286em;
}
.fa-border {
  padding: .2em .25em .15em;
  border: solid 0.08em #eee;
  border-radius: .1em;
}
.fa-pull-left {
  float: left;
}
.fa-pull-right {
  float: right;
}
.fa.fa-pull-left {
  margin-right: .3em;
}
.fa.fa-pull-right {
  margin-left: .3em;
}
/* Deprecated as of 4.4.0 */
.pull-right {
  float: right;
}
.pull-left {
  float: left;
}
.fa.pull-left {
  margin-right: .3em;
}
.fa.pull-right {
  margin-left: .3em;
}
.fa-spin {
  -webkit-animation: fa-spin 2s infinite linear;
  animation: fa-spin 2s infinite linear;
}
.fa-pulse {
  -webkit-animation: fa-spin 1s infinite steps(8);
  animation: fa-spin 1s infinite steps(8);
}
@-webkit-keyframes fa-spin {
  0% {
    -webkit-transform: rotate(0deg);
    transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(359deg);
    transform: rotate(359deg);
  }
}
@keyframes fa-spin {
  0% {
    -webkit-transform: rotate(0deg);
    transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(359deg);
    transform: rotate(359deg);
  }
}
.fa-rotate-90 {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=1)";
  -webkit-transform: rotate(90deg);
  -ms-transform: rotate(90deg);
  transform: rotate(90deg);
}
.fa-rotate-180 {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=2)";
  -webkit-transform: rotate(180deg);
  -ms-transform: rotate(180deg);
  transform: rotate(180deg);
}
.fa-rotate-270 {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=3)";
  -webkit-transform: rotate(270deg);
  -ms-transform: rotate(270deg);
  transform: rotate(270deg);
}
.fa-flip-horizontal {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=0, mirror=1)";
  -webkit-transform: scale(-1, 1);
  -ms-transform: scale(-1, 1);
  transform: scale(-1, 1);
}
.fa-flip-vertical {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=2, mirror=1)";
  -webkit-transform: scale(1, -1);
  -ms-transform: scale(1, -1);
  transform: scale(1, -1);
}
:root .fa-rotate-90,
:root .fa-rotate-180,
:root .fa-rotate-270,
:root .fa-flip-horizontal,
:root .fa-flip-vertical {
  filter: none;
}
.fa-stack {
  position: relative;
  display: inline-block;
  width: 2em;
  height: 2em;
  line-height: 2em;
  vertical-align: middle;
}
.fa-stack-1x,
.fa-stack-2x {
  position: absolute;
  left: 0;
  width: 100%;
  text-align: center;
}
.fa-stack-1x {
  line-height: inherit;
}
.fa-stack-2x {
  font-size: 2em;
}
.fa-inverse {
  color: #fff;
}
/* Font Awesome uses the Unicode Private Use Area (PUA) to ensure screen
   readers do not read off random characters that represent icons */
.fa-glass:before {
  content: "\f000";
}
.fa-music:before {
  content: "\f001";
}
.fa-search:before {
  content: "\f002";
}
.fa-envelope-o:before {
  content: "\f003";
}
.fa-heart:before {
  content: "\f004";
}
.fa-star:before {
  content: "\f005";
}
.fa-star-o:before {
  content: "\f006";
}
.fa-user:before {
  content: "\f007";
}
.fa-film:before {
  content: "\f008";
}
.fa-th-large:before {
  content: "\f009";
}
.fa-th:before {
  content: "\f00a";
}
.fa-th-list:before {
  content: "\f00b";
}
.fa-check:before {
  content: "\f00c";
}
.fa-remove:before,
.fa-close:before,
.fa-times:before {
  content: "\f00d";
}
.fa-search-plus:before {
  content: "\f00e";
}
.fa-search-minus:before {
  content: "\f010";
}
.fa-power-off:before {
  content: "\f011";
}
.fa-signal:before {
  content: "\f012";
}
.fa-gear:before,
.fa-cog:before {
  content: "\f013";
}
.fa-trash-o:before {
  content: "\f014";
}
.fa-home:before {
  content: "\f015";
}
.fa-file-o:before {
  content: "\f016";
}
.fa-clock-o:before {
  content: "\f017";
}
.fa-road:before {
  content: "\f018";
}
.fa-download:before {
  content: "\f019";
}
.fa-arrow-circle-o-down:before {
  content: "\f01a";
}
.fa-arrow-circle-o-up:before {
  content: "\f01b";
}
.fa-inbox:before {
  content: "\f01c";
}
.fa-play-circle-o:before {
  content: "\f01d";
}
.fa-rotate-right:before,
.fa-repeat:before {
  content: "\f01e";
}
.fa-refresh:before {
  content: "\f021";
}
.fa-list-alt:before {
  content: "\f022";
}
.fa-lock:before {
  content: "\f023";
}
.fa-flag:before {
  content: "\f024";
}
.fa-headphones:before {
  content: "\f025";
}
.fa-volume-off:before {
  content: "\f026";
}
.fa-volume-down:before {
  content: "\f027";
}
.fa-volume-up:before {
  content: "\f028";
}
.fa-qrcode:before {
  content: "\f029";
}
.fa-barcode:before {
  content: "\f02a";
}
.fa-tag:before {
  content: "\f02b";
}
.fa-tags:before {
  content: "\f02c";
}
.fa-book:before {
  content: "\f02d";
}
.fa-bookmark:before {
  content: "\f02e";
}
.fa-print:before {
  content: "\f02f";
}
.fa-camera:before {
  content: "\f030";
}
.fa-font:before {
  content: "\f031";
}
.fa-bold:before {
  content: "\f032";
}
.fa-italic:before {
  content: "\f033";
}
.fa-text-height:before {
  content: "\f034";
}
.fa-text-width:before {
  content: "\f035";
}
.fa-align-left:before {
  content: "\f036";
}
.fa-align-center:before {
  content: "\f037";
}
.fa-align-right:before {
  content: "\f038";
}
.fa-align-justify:before {
  content: "\f039";
}
.fa-list:before {
  content: "\f03a";
}
.fa-dedent:before,
.fa-outdent:before {
  content: "\f03b";
}
.fa-indent:before {
  content: "\f03c";
}
.fa-video-camera:before {
  content: "\f03d";
}
.fa-photo:before,
.fa-image:before,
.fa-picture-o:before {
  content: "\f03e";
}
.fa-pencil:before {
  content: "\f040";
}
.fa-map-marker:before {
  content: "\f041";
}
.fa-adjust:before {
  content: "\f042";
}
.fa-tint:before {
  content: "\f043";
}
.fa-edit:before,
.fa-pencil-square-o:before {
  content: "\f044";
}
.fa-share-square-o:before {
  content: "\f045";
}
.fa-check-square-o:before {
  content: "\f046";
}
.fa-arrows:before {
  content: "\f047";
}
.fa-step-backward:before {
  content: "\f048";
}
.fa-fast-backward:before {
  content: "\f049";
}
.fa-backward:before {
  content: "\f04a";
}
.fa-play:before {
  content: "\f04b";
}
.fa-pause:before {
  content: "\f04c";
}
.fa-stop:before {
  content: "\f04d";
}
.fa-forward:before {
  content: "\f04e";
}
.fa-fast-forward:before {
  content: "\f050";
}
.fa-step-forward:before {
  content: "\f051";
}
.fa-eject:before {
  content: "\f052";
}
.fa-chevron-left:before {
  content: "\f053";
}
.fa-chevron-right:before {
  content: "\f054";
}
.fa-plus-circle:before {
  content: "\f055";
}
.fa-minus-circle:before {
  content: "\f056";
}
.fa-times-circle:before {
  content: "\f057";
}
.fa-check-circle:before {
  content: "\f058";
}
.fa-question-circle:before {
  content: "\f059";
}
.fa-info-circle:before {
  content: "\f05a";
}
.fa-crosshairs:before {
  content: "\f05b";
}
.fa-times-circle-o:before {
  content: "\f05c";
}
.fa-check-circle-o:before {
  content: "\f05d";
}
.fa-ban:before {
  content: "\f05e";
}
.fa-arrow-left:before {
  content: "\f060";
}
.fa-arrow-right:before {
  content: "\f061";
}
.fa-arrow-up:before {
  content: "\f062";
}
.fa-arrow-down:before {
  content: "\f063";
}
.fa-mail-forward:before,
.fa-share:before {
  content: "\f064";
}
.fa-expand:before {
  content: "\f065";
}
.fa-compress:before {
  content: "\f066";
}
.fa-plus:before {
  content: "\f067";
}
.fa-minus:before {
  content: "\f068";
}
.fa-asterisk:before {
  content: "\f069";
}
.fa-exclamation-circle:before {
  content: "\f06a";
}
.fa-gift:before {
  content: "\f06b";
}
.fa-leaf:before {
  content: "\f06c";
}
.fa-fire:before {
  content: "\f06d";
}
.fa-eye:before {
  content: "\f06e";
}
.fa-eye-slash:before {
  content: "\f070";
}
.fa-warning:before,
.fa-exclamation-triangle:before {
  content: "\f071";
}
.fa-plane:before {
  content: "\f072";
}
.fa-calendar:before {
  content: "\f073";
}
.fa-random:before {
  content: "\f074";
}
.fa-comment:before {
  content: "\f075";
}
.fa-magnet:before {
  content: "\f076";
}
.fa-chevron-up:before {
  content: "\f077";
}
.fa-chevron-down:before {
  content: "\f078";
}
.fa-retweet:before {
  content: "\f079";
}
.fa-shopping-cart:before {
  content: "\f07a";
}
.fa-folder:before {
  content: "\f07b";
}
.fa-folder-open:before {
  content: "\f07c";
}
.fa-arrows-v:before {
  content: "\f07d";
}
.fa-arrows-h:before {
  content: "\f07e";
}
.fa-bar-chart-o:before,
.fa-bar-chart:before {
  content: "\f080";
}
.fa-twitter-square:before {
  content: "\f081";
}
.fa-facebook-square:before {
  content: "\f082";
}
.fa-camera-retro:before {
  content: "\f083";
}
.fa-key:before {
  content: "\f084";
}
.fa-gears:before,
.fa-cogs:before {
  content: "\f085";
}
.fa-comments:before {
  content: "\f086";
}
.fa-thumbs-o-up:before {
  content: "\f087";
}
.fa-thumbs-o-down:before {
  content: "\f088";
}
.fa-star-half:before {
  content: "\f089";
}
.fa-heart-o:before {
  content: "\f08a";
}
.fa-sign-out:before {
  content: "\f08b";
}
.fa-linkedin-square:before {
  content: "\f08c";
}
.fa-thumb-tack:before {
  content: "\f08d";
}
.fa-external-link:before {
  content: "\f08e";
}
.fa-sign-in:before {
  content: "\f090";
}
.fa-trophy:before {
  content: "\f091";
}
.fa-github-square:before {
  content: "\f092";
}
.fa-upload:before {
  content: "\f093";
}
.fa-lemon-o:before {
  content: "\f094";
}
.fa-phone:before {
  content: "\f095";
}
.fa-square-o:before {
  content: "\f096";
}
.fa-bookmark-o:before {
  content: "\f097";
}
.fa-phone-square:before {
  content: "\f098";
}
.fa-twitter:before {
  content: "\f099";
}
.fa-facebook-f:before,
.fa-facebook:before {
  content: "\f09a";
}
.fa-github:before {
  content: "\f09b";
}
.fa-unlock:before {
  content: "\f09c";
}
.fa-credit-card:before {
  content: "\f09d";
}
.fa-feed:before,
.fa-rss:before {
  content: "\f09e";
}
.fa-hdd-o:before {
  content: "\f0a0";
}
.fa-bullhorn:before {
  content: "\f0a1";
}
.fa-bell:before {
  content: "\f0f3";
}
.fa-certificate:before {
  content: "\f0a3";
}
.fa-hand-o-right:before {
  content: "\f0a4";
}
.fa-hand-o-left:before {
  content: "\f0a5";
}
.fa-hand-o-up:before {
  content: "\f0a6";
}
.fa-hand-o-down:before {
  content: "\f0a7";
}
.fa-arrow-circle-left:before {
  content: "\f0a8";
}
.fa-arrow-circle-right:before {
  content: "\f0a9";
}
.fa-arrow-circle-up:before {
  content: "\f0aa";
}
.fa-arrow-circle-down:before {
  content: "\f0ab";
}
.fa-globe:before {
  content: "\f0ac";
}
.fa-wrench:before {
  content: "\f0ad";
}
.fa-tasks:before {
  content: "\f0ae";
}
.fa-filter:before {
  content: "\f0b0";
}
.fa-briefcase:before {
  content: "\f0b1";
}
.fa-arrows-alt:before {
  content: "\f0b2";
}
.fa-group:before,
.fa-users:before {
  content: "\f0c0";
}
.fa-chain:before,
.fa-link:before {
  content: "\f0c1";
}
.fa-cloud:before {
  content: "\f0c2";
}
.fa-flask:before {
  content: "\f0c3";
}
.fa-cut:before,
.fa-scissors:before {
  content: "\f0c4";
}
.fa-copy:before,
.fa-files-o:before {
  content: "\f0c5";
}
.fa-paperclip:before {
  content: "\f0c6";
}
.fa-save:before,
.fa-floppy-o:before {
  content: "\f0c7";
}
.fa-square:before {
  content: "\f0c8";
}
.fa-navicon:before,
.fa-reorder:before,
.fa-bars:before {
  content: "\f0c9";
}
.fa-list-ul:before {
  content: "\f0ca";
}
.fa-list-ol:before {
  content: "\f0cb";
}
.fa-strikethrough:before {
  content: "\f0cc";
}
.fa-underline:before {
  content: "\f0cd";
}
.fa-table:before {
  content: "\f0ce";
}
.fa-magic:before {
  content: "\f0d0";
}
.fa-truck:before {
  content: "\f0d1";
}
.fa-pinterest:before {
  content: "\f0d2";
}
.fa-pinterest-square:before {
  content: "\f0d3";
}
.fa-google-plus-square:before {
  content: "\f0d4";
}
.fa-google-plus:before {
  content: "\f0d5";
}
.fa-money:before {
  content: "\f0d6";
}
.fa-caret-down:before {
  content: "\f0d7";
}
.fa-caret-up:before {
  content: "\f0d8";
}
.fa-caret-left:before {
  content: "\f0d9";
}
.fa-caret-right:before {
  content: "\f0da";
}
.fa-columns:before {
  content: "\f0db";
}
.fa-unsorted:before,
.fa-sort:before {
  content: "\f0dc";
}
.fa-sort-down:before,
.fa-sort-desc:before {
  content: "\f0dd";
}
.fa-sort-up:before,
.fa-sort-asc:before {
  content: "\f0de";
}
.fa-envelope:before {
  content: "\f0e0";
}
.fa-linkedin:before {
  content: "\f0e1";
}
.fa-rotate-left:before,
.fa-undo:before {
  content: "\f0e2";
}
.fa-legal:before,
.fa-gavel:before {
  content: "\f0e3";
}
.fa-dashboard:before,
.fa-tachometer:before {
  content: "\f0e4";
}
.fa-comment-o:before {
  content: "\f0e5";
}
.fa-comments-o:before {
  content: "\f0e6";
}
.fa-flash:before,
.fa-bolt:before {
  content: "\f0e7";
}
.fa-sitemap:before {
  content: "\f0e8";
}
.fa-umbrella:before {
  content: "\f0e9";
}
.fa-paste:before,
.fa-clipboard:before {
  content: "\f0ea";
}
.fa-lightbulb-o:before {
  content: "\f0eb";
}
.fa-exchange:before {
  content: "\f0ec";
}
.fa-cloud-download:before {
  content: "\f0ed";
}
.fa-cloud-upload:before {
  content: "\f0ee";
}
.fa-user-md:before {
  content: "\f0f0";
}
.fa-stethoscope:before {
  content: "\f0f1";
}
.fa-suitcase:before {
  content: "\f0f2";
}
.fa-bell-o:before {
  content: "\f0a2";
}
.fa-coffee:before {
  content: "\f0f4";
}
.fa-cutlery:before {
  content: "\f0f5";
}
.fa-file-text-o:before {
  content: "\f0f6";
}
.fa-building-o:before {
  content: "\f0f7";
}
.fa-hospital-o:before {
  content: "\f0f8";
}
.fa-ambulance:before {
  content: "\f0f9";
}
.fa-medkit:before {
  content: "\f0fa";
}
.fa-fighter-jet:before {
  content: "\f0fb";
}
.fa-beer:before {
  content: "\f0fc";
}
.fa-h-square:before {
  content: "\f0fd";
}
.fa-plus-square:before {
  content: "\f0fe";
}
.fa-angle-double-left:before {
  content: "\f100";
}
.fa-angle-double-right:before {
  content: "\f101";
}
.fa-angle-double-up:before {
  content: "\f102";
}
.fa-angle-double-down:before {
  content: "\f103";
}
.fa-angle-left:before {
  content: "\f104";
}
.fa-angle-right:before {
  content: "\f105";
}
.fa-angle-up:before {
  content: "\f106";
}
.fa-angle-down:before {
  content: "\f107";
}
.fa-desktop:before {
  content: "\f108";
}
.fa-laptop:before {
  content: "\f109";
}
.fa-tablet:before {
  content: "\f10a";
}
.fa-mobile-phone:before,
.fa-mobile:before {
  content: "\f10b";
}
.fa-circle-o:before {
  content: "\f10c";
}
.fa-quote-left:before {
  content: "\f10d";
}
.fa-quote-right:before {
  content: "\f10e";
}
.fa-spinner:before {
  content: "\f110";
}
.fa-circle:before {
  content: "\f111";
}
.fa-mail-reply:before,
.fa-reply:before {
  content: "\f112";
}
.fa-github-alt:before {
  content: "\f113";
}
.fa-folder-o:before {
  content: "\f114";
}
.fa-folder-open-o:before {
  content: "\f115";
}
.fa-smile-o:before {
  content: "\f118";
}
.fa-frown-o:before {
  content: "\f119";
}
.fa-meh-o:before {
  content: "\f11a";
}
.fa-gamepad:before {
  content: "\f11b";
}
.fa-keyboard-o:before {
  content: "\f11c";
}
.fa-flag-o:before {
  content: "\f11d";
}
.fa-flag-checkered:before {
  content: "\f11e";
}
.fa-terminal:before {
  content: "\f120";
}
.fa-code:before {
  content: "\f121";
}
.fa-mail-reply-all:before,
.fa-reply-all:before {
  content: "\f122";
}
.fa-star-half-empty:before,
.fa-star-half-full:before,
.fa-star-half-o:before {
  content: "\f123";
}
.fa-location-arrow:before {
  content: "\f124";
}
.fa-crop:before {
  content: "\f125";
}
.fa-code-fork:before {
  content: "\f126";
}
.fa-unlink:before,
.fa-chain-broken:before {
  content: "\f127";
}
.fa-question:before {
  content: "\f128";
}
.fa-info:before {
  content: "\f129";
}
.fa-exclamation:before {
  content: "\f12a";
}
.fa-superscript:before {
  content: "\f12b";
}
.fa-subscript:before {
  content: "\f12c";
}
.fa-eraser:before {
  content: "\f12d";
}
.fa-puzzle-piece:before {
  content: "\f12e";
}
.fa-microphone:before {
  content: "\f130";
}
.fa-microphone-slash:before {
  content: "\f131";
}
.fa-shield:before {
  content: "\f132";
}
.fa-calendar-o:before {
  content: "\f133";
}
.fa-fire-extinguisher:before {
  content: "\f134";
}
.fa-rocket:before {
  content: "\f135";
}
.fa-maxcdn:before {
  content: "\f136";
}
.fa-chevron-circle-left:before {
  content: "\f137";
}
.fa-chevron-circle-right:before {
  content: "\f138";
}
.fa-chevron-circle-up:before {
  content: "\f139";
}
.fa-chevron-circle-down:before {
  content: "\f13a";
}
.fa-html5:before {
  content: "\f13b";
}
.fa-css3:before {
  content: "\f13c";
}
.fa-anchor:before {
  content: "\f13d";
}
.fa-unlock-alt:before {
  content: "\f13e";
}
.fa-bullseye:before {
  content: "\f140";
}
.fa-ellipsis-h:before {
  content: "\f141";
}
.fa-ellipsis-v:before {
  content: "\f142";
}
.fa-rss-square:before {
  content: "\f143";
}
.fa-play-circle:before {
  content: "\f144";
}
.fa-ticket:before {
  content: "\f145";
}
.fa-minus-square:before {
  content: "\f146";
}
.fa-minus-square-o:before {
  content: "\f147";
}
.fa-level-up:before {
  content: "\f148";
}
.fa-level-down:before {
  content: "\f149";
}
.fa-check-square:before {
  content: "\f14a";
}
.fa-pencil-square:before {
  content: "\f14b";
}
.fa-external-link-square:before {
  content: "\f14c";
}
.fa-share-square:before {
  content: "\f14d";
}
.fa-compass:before {
  content: "\f14e";
}
.fa-toggle-down:before,
.fa-caret-square-o-down:before {
  content: "\f150";
}
.fa-toggle-up:before,
.fa-caret-square-o-up:before {
  content: "\f151";
}
.fa-toggle-right:before,
.fa-caret-square-o-right:before {
  content: "\f152";
}
.fa-euro:before,
.fa-eur:before {
  content: "\f153";
}
.fa-gbp:before {
  content: "\f154";
}
.fa-dollar:before,
.fa-usd:before {
  content: "\f155";
}
.fa-rupee:before,
.fa-inr:before {
  content: "\f156";
}
.fa-cny:before,
.fa-rmb:before,
.fa-yen:before,
.fa-jpy:before {
  content: "\f157";
}
.fa-ruble:before,
.fa-rouble:before,
.fa-rub:before {
  content: "\f158";
}
.fa-won:before,
.fa-krw:before {
  content: "\f159";
}
.fa-bitcoin:before,
.fa-btc:before {
  content: "\f15a";
}
.fa-file:before {
  content: "\f15b";
}
.fa-file-text:before {
  content: "\f15c";
}
.fa-sort-alpha-asc:before {
  content: "\f15d";
}
.fa-sort-alpha-desc:before {
  content: "\f15e";
}
.fa-sort-amount-asc:before {
  content: "\f160";
}
.fa-sort-amount-desc:before {
  content: "\f161";
}
.fa-sort-numeric-asc:before {
  content: "\f162";
}
.fa-sort-numeric-desc:before {
  content: "\f163";
}
.fa-thumbs-up:before {
  content: "\f164";
}
.fa-thumbs-down:before {
  content: "\f165";
}
.fa-youtube-square:before {
  content: "\f166";
}
.fa-youtube:before {
  content: "\f167";
}
.fa-xing:before {
  content: "\f168";
}
.fa-xing-square:before {
  content: "\f169";
}
.fa-youtube-play:before {
  content: "\f16a";
}
.fa-dropbox:before {
  content: "\f16b";
}
.fa-stack-overflow:before {
  content: "\f16c";
}
.fa-instagram:before {
  content: "\f16d";
}
.fa-flickr:before {
  content: "\f16e";
}
.fa-adn:before {
  content: "\f170";
}
.fa-bitbucket:before {
  content: "\f171";
}
.fa-bitbucket-square:before {
  content: "\f172";
}
.fa-tumblr:before {
  content: "\f173";
}
.fa-tumblr-square:before {
  content: "\f174";
}
.fa-long-arrow-down:before {
  content: "\f175";
}
.fa-long-arrow-up:before {
  content: "\f176";
}
.fa-long-arrow-left:before {
  content: "\f177";
}
.fa-long-arrow-right:before {
  content: "\f178";
}
.fa-apple:before {
  content: "\f179";
}
.fa-windows:before {
  content: "\f17a";
}
.fa-android:before {
  content: "\f17b";
}
.fa-linux:before {
  content: "\f17c";
}
.fa-dribbble:before {
  content: "\f17d";
}
.fa-skype:before {
  content: "\f17e";
}
.fa-foursquare:before {
  content: "\f180";
}
.fa-trello:before {
  content: "\f181";
}
.fa-female:before {
  content: "\f182";
}
.fa-male:before {
  content: "\f183";
}
.fa-gittip:before,
.fa-gratipay:before {
  content: "\f184";
}
.fa-sun-o:before {
  content: "\f185";
}
.fa-moon-o:before {
  content: "\f186";
}
.fa-archive:before {
  content: "\f187";
}
.fa-bug:before {
  content: "\f188";
}
.fa-vk:before {
  content: "\f189";
}
.fa-weibo:before {
  content: "\f18a";
}
.fa-renren:before {
  content: "\f18b";
}
.fa-pagelines:before {
  content: "\f18c";
}
.fa-stack-exchange:before {
  content: "\f18d";
}
.fa-arrow-circle-o-right:before {
  content: "\f18e";
}
.fa-arrow-circle-o-left:before {
  content: "\f190";
}
.fa-toggle-left:before,
.fa-caret-square-o-left:before {
  content: "\f191";
}
.fa-dot-circle-o:before {
  content: "\f192";
}
.fa-wheelchair:before {
  content: "\f193";
}
.fa-vimeo-square:before {
  content: "\f194";
}
.fa-turkish-lira:before,
.fa-try:before {
  content: "\f195";
}
.fa-plus-square-o:before {
  content: "\f196";
}
.fa-space-shuttle:before {
  content: "\f197";
}
.fa-slack:before {
  content: "\f198";
}
.fa-envelope-square:before {
  content: "\f199";
}
.fa-wordpress:before {
  content: "\f19a";
}
.fa-openid:before {
  content: "\f19b";
}
.fa-institution:before,
.fa-bank:before,
.fa-university:before {
  content: "\f19c";
}
.fa-mortar-board:before,
.fa-graduation-cap:before {
  content: "\f19d";
}
.fa-yahoo:before {
  content: "\f19e";
}
.fa-google:before {
  content: "\f1a0";
}
.fa-reddit:before {
  content: "\f1a1";
}
.fa-reddit-square:before {
  content: "\f1a2";
}
.fa-stumbleupon-circle:before {
  content: "\f1a3";
}
.fa-stumbleupon:before {
  content: "\f1a4";
}
.fa-delicious:before {
  content: "\f1a5";
}
.fa-digg:before {
  content: "\f1a6";
}
.fa-pied-piper-pp:before {
  content: "\f1a7";
}
.fa-pied-piper-alt:before {
  content: "\f1a8";
}
.fa-drupal:before {
  content: "\f1a9";
}
.fa-joomla:before {
  content: "\f1aa";
}
.fa-language:before {
  content: "\f1ab";
}
.fa-fax:before {
  content: "\f1ac";
}
.fa-building:before {
  content: "\f1ad";
}
.fa-child:before {
  content: "\f1ae";
}
.fa-paw:before {
  content: "\f1b0";
}
.fa-spoon:before {
  content: "\f1b1";
}
.fa-cube:before {
  content: "\f1b2";
}
.fa-cubes:before {
  content: "\f1b3";
}
.fa-behance:before {
  content: "\f1b4";
}
.fa-behance-square:before {
  content: "\f1b5";
}
.fa-steam:before {
  content: "\f1b6";
}
.fa-steam-square:before {
  content: "\f1b7";
}
.fa-recycle:before {
  content: "\f1b8";
}
.fa-automobile:before,
.fa-car:before {
  content: "\f1b9";
}
.fa-cab:before,
.fa-taxi:before {
  content: "\f1ba";
}
.fa-tree:before {
  content: "\f1bb";
}
.fa-spotify:before {
  content: "\f1bc";
}
.fa-deviantart:before {
  content: "\f1bd";
}
.fa-soundcloud:before {
  content: "\f1be";
}
.fa-database:before {
  content: "\f1c0";
}
.fa-file-pdf-o:before {
  content: "\f1c1";
}
.fa-file-word-o:before {
  content: "\f1c2";
}
.fa-file-excel-o:before {
  content: "\f1c3";
}
.fa-file-powerpoint-o:before {
  content: "\f1c4";
}
.fa-file-photo-o:before,
.fa-file-picture-o:before,
.fa-file-image-o:before {
  content: "\f1c5";
}
.fa-file-zip-o:before,
.fa-file-archive-o:before {
  content: "\f1c6";
}
.fa-file-sound-o:before,
.fa-file-audio-o:before {
  content: "\f1c7";
}
.fa-file-movie-o:before,
.fa-file-video-o:before {
  content: "\f1c8";
}
.fa-file-code-o:before {
  content: "\f1c9";
}
.fa-vine:before {
  content: "\f1ca";
}
.fa-codepen:before {
  content: "\f1cb";
}
.fa-jsfiddle:before {
  content: "\f1cc";
}
.fa-life-bouy:before,
.fa-life-buoy:before,
.fa-life-saver:before,
.fa-support:before,
.fa-life-ring:before {
  content: "\f1cd";
}
.fa-circle-o-notch:before {
  content: "\f1ce";
}
.fa-ra:before,
.fa-resistance:before,
.fa-rebel:before {
  content: "\f1d0";
}
.fa-ge:before,
.fa-empire:before {
  content: "\f1d1";
}
.fa-git-square:before {
  content: "\f1d2";
}
.fa-git:before {
  content: "\f1d3";
}
.fa-y-combinator-square:before,
.fa-yc-square:before,
.fa-hacker-news:before {
  content: "\f1d4";
}
.fa-tencent-weibo:before {
  content: "\f1d5";
}
.fa-qq:before {
  content: "\f1d6";
}
.fa-wechat:before,
.fa-weixin:before {
  content: "\f1d7";
}
.fa-send:before,
.fa-paper-plane:before {
  content: "\f1d8";
}
.fa-send-o:before,
.fa-paper-plane-o:before {
  content: "\f1d9";
}
.fa-history:before {
  content: "\f1da";
}
.fa-circle-thin:before {
  content: "\f1db";
}
.fa-header:before {
  content: "\f1dc";
}
.fa-paragraph:before {
  content: "\f1dd";
}
.fa-sliders:before {
  content: "\f1de";
}
.fa-share-alt:before {
  content: "\f1e0";
}
.fa-share-alt-square:before {
  content: "\f1e1";
}
.fa-bomb:before {
  content: "\f1e2";
}
.fa-soccer-ball-o:before,
.fa-futbol-o:before {
  content: "\f1e3";
}
.fa-tty:before {
  content: "\f1e4";
}
.fa-binoculars:before {
  content: "\f1e5";
}
.fa-plug:before {
  content: "\f1e6";
}
.fa-slideshare:before {
  content: "\f1e7";
}
.fa-twitch:before {
  content: "\f1e8";
}
.fa-yelp:before {
  content: "\f1e9";
}
.fa-newspaper-o:before {
  content: "\f1ea";
}
.fa-wifi:before {
  content: "\f1eb";
}
.fa-calculator:before {
  content: "\f1ec";
}
.fa-paypal:before {
  content: "\f1ed";
}
.fa-google-wallet:before {
  content: "\f1ee";
}
.fa-cc-visa:before {
  content: "\f1f0";
}
.fa-cc-mastercard:before {
  content: "\f1f1";
}
.fa-cc-discover:before {
  content: "\f1f2";
}
.fa-cc-amex:before {
  content: "\f1f3";
}
.fa-cc-paypal:before {
  content: "\f1f4";
}
.fa-cc-stripe:before {
  content: "\f1f5";
}
.fa-bell-slash:before {
  content: "\f1f6";
}
.fa-bell-slash-o:before {
  content: "\f1f7";
}
.fa-trash:before {
  content: "\f1f8";
}
.fa-copyright:before {
  content: "\f1f9";
}
.fa-at:before {
  content: "\f1fa";
}
.fa-eyedropper:before {
  content: "\f1fb";
}
.fa-paint-brush:before {
  content: "\f1fc";
}
.fa-birthday-cake:before {
  content: "\f1fd";
}
.fa-area-chart:before {
  content: "\f1fe";
}
.fa-pie-chart:before {
  content: "\f200";
}
.fa-line-chart:before {
  content: "\f201";
}
.fa-lastfm:before {
  content: "\f202";
}
.fa-lastfm-square:before {
  content: "\f203";
}
.fa-toggle-off:before {
  content: "\f204";
}
.fa-toggle-on:before {
  content: "\f205";
}
.fa-bicycle:before {
  content: "\f206";
}
.fa-bus:before {
  content: "\f207";
}
.fa-ioxhost:before {
  content: "\f208";
}
.fa-angellist:before {
  content: "\f209";
}
.fa-cc:before {
  content: "\f20a";
}
.fa-shekel:before,
.fa-sheqel:before,
.fa-ils:before {
  content: "\f20b";
}
.fa-meanpath:before {
  content: "\f20c";
}
.fa-buysellads:before {
  content: "\f20d";
}
.fa-connectdevelop:before {
  content: "\f20e";
}
.fa-dashcube:before {
  content: "\f210";
}
.fa-forumbee:before {
  content: "\f211";
}
.fa-leanpub:before {
  content: "\f212";
}
.fa-sellsy:before {
  content: "\f213";
}
.fa-shirtsinbulk:before {
  content: "\f214";
}
.fa-simplybuilt:before {
  content: "\f215";
}
.fa-skyatlas:before {
  content: "\f216";
}
.fa-cart-plus:before {
  content: "\f217";
}
.fa-cart-arrow-down:before {
  content: "\f218";
}
.fa-diamond:before {
  content: "\f219";
}
.fa-ship:before {
  content: "\f21a";
}
.fa-user-secret:before {
  content: "\f21b";
}
.fa-motorcycle:before {
  content: "\f21c";
}
.fa-street-view:before {
  content: "\f21d";
}
.fa-heartbeat:before {
  content: "\f21e";
}
.fa-venus:before {
  content: "\f221";
}
.fa-mars:before {
  content: "\f222";
}
.fa-mercury:before {
  content: "\f223";
}
.fa-intersex:before,
.fa-transgender:before {
  content: "\f224";
}
.fa-transgender-alt:before {
  content: "\f225";
}
.fa-venus-double:before {
  content: "\f226";
}
.fa-mars-double:before {
  content: "\f227";
}
.fa-venus-mars:before {
  content: "\f228";
}
.fa-mars-stroke:before {
  content: "\f229";
}
.fa-mars-stroke-v:before {
  content: "\f22a";
}
.fa-mars-stroke-h:before {
  content: "\f22b";
}
.fa-neuter:before {
  content: "\f22c";
}
.fa-genderless:before {
  content: "\f22d";
}
.fa-facebook-official:before {
  content: "\f230";
}
.fa-pinterest-p:before {
  content: "\f231";
}
.fa-whatsapp:before {
  content: "\f232";
}
.fa-server:before {
  content: "\f233";
}
.fa-user-plus:before {
  content: "\f234";
}
.fa-user-times:before {
  content: "\f235";
}
.fa-hotel:before,
.fa-bed:before {
  content: "\f236";
}
.fa-viacoin:before {
  content: "\f237";
}
.fa-train:before {
  content: "\f238";
}
.fa-subway:before {
  content: "\f239";
}
.fa-medium:before {
  content: "\f23a";
}
.fa-yc:before,
.fa-y-combinator:before {
  content: "\f23b";
}
.fa-optin-monster:before {
  content: "\f23c";
}
.fa-opencart:before {
  content: "\f23d";
}
.fa-expeditedssl:before {
  content: "\f23e";
}
.fa-battery-4:before,
.fa-battery:before,
.fa-battery-full:before {
  content: "\f240";
}
.fa-battery-3:before,
.fa-battery-three-quarters:before {
  content: "\f241";
}
.fa-battery-2:before,
.fa-battery-half:before {
  content: "\f242";
}
.fa-battery-1:before,
.fa-battery-quarter:before {
  content: "\f243";
}
.fa-battery-0:before,
.fa-battery-empty:before {
  content: "\f244";
}
.fa-mouse-pointer:before {
  content: "\f245";
}
.fa-i-cursor:before {
  content: "\f246";
}
.fa-object-group:before {
  content: "\f247";
}
.fa-object-ungroup:before {
  content: "\f248";
}
.fa-sticky-note:before {
  content: "\f249";
}
.fa-sticky-note-o:before {
  content: "\f24a";
}
.fa-cc-jcb:before {
  content: "\f24b";
}
.fa-cc-diners-club:before {
  content: "\f24c";
}
.fa-clone:before {
  content: "\f24d";
}
.fa-balance-scale:before {
  content: "\f24e";
}
.fa-hourglass-o:before {
  content: "\f250";
}
.fa-hourglass-1:before,
.fa-hourglass-start:before {
  content: "\f251";
}
.fa-hourglass-2:before,
.fa-hourglass-half:before {
  content: "\f252";
}
.fa-hourglass-3:before,
.fa-hourglass-end:before {
  content: "\f253";
}
.fa-hourglass:before {
  content: "\f254";
}
.fa-hand-grab-o:before,
.fa-hand-rock-o:before {
  content: "\f255";
}
.fa-hand-stop-o:before,
.fa-hand-paper-o:before {
  content: "\f256";
}
.fa-hand-scissors-o:before {
  content: "\f257";
}
.fa-hand-lizard-o:before {
  content: "\f258";
}
.fa-hand-spock-o:before {
  content: "\f259";
}
.fa-hand-pointer-o:before {
  content: "\f25a";
}
.fa-hand-peace-o:before {
  content: "\f25b";
}
.fa-trademark:before {
  content: "\f25c";
}
.fa-registered:before {
  content: "\f25d";
}
.fa-creative-commons:before {
  content: "\f25e";
}
.fa-gg:before {
  content: "\f260";
}
.fa-gg-circle:before {
  content: "\f261";
}
.fa-tripadvisor:before {
  content: "\f262";
}
.fa-odnoklassniki:before {
  content: "\f263";
}
.fa-odnoklassniki-square:before {
  content: "\f264";
}
.fa-get-pocket:before {
  content: "\f265";
}
.fa-wikipedia-w:before {
  content: "\f266";
}
.fa-safari:before {
  content: "\f267";
}
.fa-chrome:before {
  content: "\f268";
}
.fa-firefox:before {
  content: "\f269";
}
.fa-opera:before {
  content: "\f26a";
}
.fa-internet-explorer:before {
  content: "\f26b";
}
.fa-tv:before,
.fa-television:before {
  content: "\f26c";
}
.fa-contao:before {
  content: "\f26d";
}
.fa-500px:before {
  content: "\f26e";
}
.fa-amazon:before {
  content: "\f270";
}
.fa-calendar-plus-o:before {
  content: "\f271";
}
.fa-calendar-minus-o:before {
  content: "\f272";
}
.fa-calendar-times-o:before {
  content: "\f273";
}
.fa-calendar-check-o:before {
  content: "\f274";
}
.fa-industry:before {
  content: "\f275";
}
.fa-map-pin:before {
  content: "\f276";
}
.fa-map-signs:before {
  content: "\f277";
}
.fa-map-o:before {
  content: "\f278";
}
.fa-map:before {
  content: "\f279";
}
.fa-commenting:before {
  content: "\f27a";
}
.fa-commenting-o:before {
  content: "\f27b";
}
.fa-houzz:before {
  content: "\f27c";
}
.fa-vimeo:before {
  content: "\f27d";
}
.fa-black-tie:before {
  content: "\f27e";
}
.fa-fonticons:before {
  content: "\f280";
}
.fa-reddit-alien:before {
  content: "\f281";
}
.fa-edge:before {
  content: "\f282";
}
.fa-credit-card-alt:before {
  content: "\f283";
}
.fa-codiepie:before {
  content: "\f284";
}
.fa-modx:before {
  content: "\f285";
}
.fa-fort-awesome:before {
  content: "\f286";
}
.fa-usb:before {
  content: "\f287";
}
.fa-product-hunt:before {
  content: "\f288";
}
.fa-mixcloud:before {
  content: "\f289";
}
.fa-scribd:before {
  content: "\f28a";
}
.fa-pause-circle:before {
  content: "\f28b";
}
.fa-pause-circle-o:before {
  content: "\f28c";
}
.fa-stop-circle:before {
  content: "\f28d";
}
.fa-stop-circle-o:before {
  content: "\f28e";
}
.fa-shopping-bag:before {
  content: "\f290";
}
.fa-shopping-basket:before {
  content: "\f291";
}
.fa-hashtag:before {
  content: "\f292";
}
.fa-bluetooth:before {
  content: "\f293";
}
.fa-bluetooth-b:before {
  content: "\f294";
}
.fa-percent:before {
  content: "\f295";
}
.fa-gitlab:before {
  content: "\f296";
}
.fa-wpbeginner:before {
  content: "\f297";
}
.fa-wpforms:before {
  content: "\f298";
}
.fa-envira:before {
  content: "\f299";
}
.fa-universal-access:before {
  content: "\f29a";
}
.fa-wheelchair-alt:before {
  content: "\f29b";
}
.fa-question-circle-o:before {
  content: "\f29c";
}
.fa-blind:before {
  content: "\f29d";
}
.fa-audio-description:before {
  content: "\f29e";
}
.fa-volume-control-phone:before {
  content: "\f2a0";
}
.fa-braille:before {
  content: "\f2a1";
}
.fa-assistive-listening-systems:before {
  content: "\f2a2";
}
.fa-asl-interpreting:before,
.fa-american-sign-language-interpreting:before {
  content: "\f2a3";
}
.fa-deafness:before,
.fa-hard-of-hearing:before,
.fa-deaf:before {
  content: "\f2a4";
}
.fa-glide:before {
  content: "\f2a5";
}
.fa-glide-g:before {
  content: "\f2a6";
}
.fa-signing:before,
.fa-sign-language:before {
  content: "\f2a7";
}
.fa-low-vision:before {
  content: "\f2a8";
}
.fa-viadeo:before {
  content: "\f2a9";
}
.fa-viadeo-square:before {
  content: "\f2aa";
}
.fa-snapchat:before {
  content: "\f2ab";
}
.fa-snapchat-ghost:before {
  content: "\f2ac";
}
.fa-snapchat-square:before {
  content: "\f2ad";
}
.fa-pied-piper:before {
  content: "\f2ae";
}
.fa-first-order:before {
  content: "\f2b0";
}
.fa-yoast:before {
  content: "\f2b1";
}
.fa-themeisle:before {
  content: "\f2b2";
}
.fa-google-plus-circle:before,
.fa-google-plus-official:before {
  content: "\f2b3";
}
.fa-fa:before,
.fa-font-awesome:before {
  content: "\f2b4";
}
.fa-handshake-o:before {
  content: "\f2b5";
}
.fa-envelope-open:before {
  content: "\f2b6";
}
.fa-envelope-open-o:before {
  content: "\f2b7";
}
.fa-linode:before {
  content: "\f2b8";
}
.fa-address-book:before {
  content: "\f2b9";
}
.fa-address-book-o:before {
  content: "\f2ba";
}
.fa-vcard:before,
.fa-address-card:before {
  content: "\f2bb";
}
.fa-vcard-o:before,
.fa-address-card-o:before {
  content: "\f2bc";
}
.fa-user-circle:before {
  content: "\f2bd";
}
.fa-user-circle-o:before {
  content: "\f2be";
}
.fa-user-o:before {
  content: "\f2c0";
}
.fa-id-badge:before {
  content: "\f2c1";
}
.fa-drivers-license:before,
.fa-id-card:before {
  content: "\f2c2";
}
.fa-drivers-license-o:before,
.fa-id-card-o:before {
  content: "\f2c3";
}
.fa-quora:before {
  content: "\f2c4";
}
.fa-free-code-camp:before {
  content: "\f2c5";
}
.fa-telegram:before {
  content: "\f2c6";
}
.fa-thermometer-4:before,
.fa-thermometer:before,
.fa-thermometer-full:before {
  content: "\f2c7";
}
.fa-thermometer-3:before,
.fa-thermometer-three-quarters:before {
  content: "\f2c8";
}
.fa-thermometer-2:before,
.fa-thermometer-half:before {
  content: "\f2c9";
}
.fa-thermometer-1:before,
.fa-thermometer-quarter:before {
  content: "\f2ca";
}
.fa-thermometer-0:before,
.fa-thermometer-empty:before {
  content: "\f2cb";
}
.fa-shower:before {
  content: "\f2cc";
}
.fa-bathtub:before,
.fa-s15:before,
.fa-bath:before {
  content: "\f2cd";
}
.fa-podcast:before {
  content: "\f2ce";
}
.fa-window-maximize:before {
  content: "\f2d0";
}
.fa-window-minimize:before {
  content: "\f2d1";
}
.fa-window-restore:before {
  content: "\f2d2";
}
.fa-times-rectangle:before,
.fa-window-close:before {
  content: "\f2d3";
}
.fa-times-rectangle-o:before,
.fa-window-close-o:before {
  content: "\f2d4";
}
.fa-bandcamp:before {
  content: "\f2d5";
}
.fa-grav:before {
  content: "\f2d6";
}
.fa-etsy:before {
  content: "\f2d7";
}
.fa-imdb:before {
  content: "\f2d8";
}
.fa-ravelry:before {
  content: "\f2d9";
}
.fa-eercast:before {
  content: "\f2da";
}
.fa-microchip:before {
  content: "\f2db";
}
.fa-snowflake-o:before {
  content: "\f2dc";
}
.fa-superpowers:before {
  content: "\f2dd";
}
.fa-wpexplorer:before {
  content: "\f2de";
}
.fa-meetup:before {
  content: "\f2e0";
}
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}
.sr-only-focusable:active,
.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  margin: 0;
  overflow: visible;
  clip: auto;
}
.sr-only-focusable:active,
.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  margin: 0;
  overflow: visible;
  clip: auto;
}
/*!
*
* IPython base
*
*/
.modal.fade .modal-dialog {
  -webkit-transform: translate(0, 0);
  -ms-transform: translate(0, 0);
  -o-transform: translate(0, 0);
  transform: translate(0, 0);
}
code {
  color: #000;
}
pre {
  font-size: inherit;
  line-height: inherit;
}
label {
  font-weight: normal;
}
/* Make the page background atleast 100% the height of the view port */
/* Make the page itself atleast 70% the height of the view port */
.border-box-sizing {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
.corner-all {
  border-radius: 2px;
}
.no-padding {
  padding: 0px;
}
/* Flexible box model classes */
/* Taken from Alex Russell http://infrequently.org/2009/08/css-3-progress/ */
/* This file is a compatability layer.  It allows the usage of flexible box 
model layouts accross multiple browsers, including older browsers.  The newest,
universal implementation of the flexible box model is used when available (see
`Modern browsers` comments below).  Browsers that are known to implement this 
new spec completely include:

    Firefox 28.0+
    Chrome 29.0+
    Internet Explorer 11+ 
    Opera 17.0+

Browsers not listed, including Safari, are supported via the styling under the
`Old browsers` comments below.
*/
.hbox {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
.hbox > * {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
}
.vbox {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
.vbox > * {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
}
.hbox.reverse,
.vbox.reverse,
.reverse {
  /* Old browsers */
  -webkit-box-direction: reverse;
  -moz-box-direction: reverse;
  box-direction: reverse;
  /* Modern browsers */
  flex-direction: row-reverse;
}
.hbox.box-flex0,
.vbox.box-flex0,
.box-flex0 {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
  width: auto;
}
.hbox.box-flex1,
.vbox.box-flex1,
.box-flex1 {
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
.hbox.box-flex,
.vbox.box-flex,
.box-flex {
  /* Old browsers */
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
.hbox.box-flex2,
.vbox.box-flex2,
.box-flex2 {
  /* Old browsers */
  -webkit-box-flex: 2;
  -moz-box-flex: 2;
  box-flex: 2;
  /* Modern browsers */
  flex: 2;
}
.box-group1 {
  /*  Deprecated */
  -webkit-box-flex-group: 1;
  -moz-box-flex-group: 1;
  box-flex-group: 1;
}
.box-group2 {
  /* Deprecated */
  -webkit-box-flex-group: 2;
  -moz-box-flex-group: 2;
  box-flex-group: 2;
}
.hbox.start,
.vbox.start,
.start {
  /* Old browsers */
  -webkit-box-pack: start;
  -moz-box-pack: start;
  box-pack: start;
  /* Modern browsers */
  justify-content: flex-start;
}
.hbox.end,
.vbox.end,
.end {
  /* Old browsers */
  -webkit-box-pack: end;
  -moz-box-pack: end;
  box-pack: end;
  /* Modern browsers */
  justify-content: flex-end;
}
.hbox.center,
.vbox.center,
.center {
  /* Old browsers */
  -webkit-box-pack: center;
  -moz-box-pack: center;
  box-pack: center;
  /* Modern browsers */
  justify-content: center;
}
.hbox.baseline,
.vbox.baseline,
.baseline {
  /* Old browsers */
  -webkit-box-pack: baseline;
  -moz-box-pack: baseline;
  box-pack: baseline;
  /* Modern browsers */
  justify-content: baseline;
}
.hbox.stretch,
.vbox.stretch,
.stretch {
  /* Old browsers */
  -webkit-box-pack: stretch;
  -moz-box-pack: stretch;
  box-pack: stretch;
  /* Modern browsers */
  justify-content: stretch;
}
.hbox.align-start,
.vbox.align-start,
.align-start {
  /* Old browsers */
  -webkit-box-align: start;
  -moz-box-align: start;
  box-align: start;
  /* Modern browsers */
  align-items: flex-start;
}
.hbox.align-end,
.vbox.align-end,
.align-end {
  /* Old browsers */
  -webkit-box-align: end;
  -moz-box-align: end;
  box-align: end;
  /* Modern browsers */
  align-items: flex-end;
}
.hbox.align-center,
.vbox.align-center,
.align-center {
  /* Old browsers */
  -webkit-box-align: center;
  -moz-box-align: center;
  box-align: center;
  /* Modern browsers */
  align-items: center;
}
.hbox.align-baseline,
.vbox.align-baseline,
.align-baseline {
  /* Old browsers */
  -webkit-box-align: baseline;
  -moz-box-align: baseline;
  box-align: baseline;
  /* Modern browsers */
  align-items: baseline;
}
.hbox.align-stretch,
.vbox.align-stretch,
.align-stretch {
  /* Old browsers */
  -webkit-box-align: stretch;
  -moz-box-align: stretch;
  box-align: stretch;
  /* Modern browsers */
  align-items: stretch;
}
div.error {
  margin: 2em;
  text-align: center;
}
div.error > h1 {
  font-size: 500%;
  line-height: normal;
}
div.error > p {
  font-size: 200%;
  line-height: normal;
}
div.traceback-wrapper {
  text-align: left;
  max-width: 800px;
  margin: auto;
}
div.traceback-wrapper pre.traceback {
  max-height: 600px;
  overflow: auto;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
body {
  background-color: #fff;
  /* This makes sure that the body covers the entire window and needs to
       be in a different element than the display: box in wrapper below */
  position: absolute;
  left: 0px;
  right: 0px;
  top: 0px;
  bottom: 0px;
  overflow: visible;
}
body > #header {
  /* Initially hidden to prevent FLOUC */
  display: none;
  background-color: #fff;
  /* Display over codemirror */
  position: relative;
  z-index: 100;
}
body > #header #header-container {
  display: flex;
  flex-direction: row;
  justify-content: space-between;
  padding: 5px;
  padding-bottom: 5px;
  padding-top: 5px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
body > #header .header-bar {
  width: 100%;
  height: 1px;
  background: #e7e7e7;
  margin-bottom: -1px;
}
@media print {
  body > #header {
    display: none !important;
  }
}
#header-spacer {
  width: 100%;
  visibility: hidden;
}
@media print {
  #header-spacer {
    display: none;
  }
}
#ipython_notebook {
  padding-left: 0px;
  padding-top: 1px;
  padding-bottom: 1px;
}
[dir="rtl"] #ipython_notebook {
  margin-right: 10px;
  margin-left: 0;
}
[dir="rtl"] #ipython_notebook.pull-left {
  float: right !important;
  float: right;
}
.flex-spacer {
  flex: 1;
}
#noscript {
  width: auto;
  padding-top: 16px;
  padding-bottom: 16px;
  text-align: center;
  font-size: 22px;
  color: red;
  font-weight: bold;
}
#ipython_notebook img {
  height: 28px;
}
#site {
  width: 100%;
  display: none;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  overflow: auto;
}
@media print {
  #site {
    height: auto !important;
  }
}
/* Smaller buttons */
.ui-button .ui-button-text {
  padding: 0.2em 0.8em;
  font-size: 77%;
}
input.ui-button {
  padding: 0.3em 0.9em;
}
span#kernel_logo_widget {
  margin: 0 10px;
}
span#login_widget {
  float: right;
}
[dir="rtl"] span#login_widget {
  float: left;
}
span#login_widget > .button,
#logout {
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
span#login_widget > .button:focus,
#logout:focus,
span#login_widget > .button.focus,
#logout.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
span#login_widget > .button:hover,
#logout:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
span#login_widget > .button:active,
#logout:active,
span#login_widget > .button.active,
#logout.active,
.open > .dropdown-togglespan#login_widget > .button,
.open > .dropdown-toggle#logout {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
span#login_widget > .button:active:hover,
#logout:active:hover,
span#login_widget > .button.active:hover,
#logout.active:hover,
.open > .dropdown-togglespan#login_widget > .button:hover,
.open > .dropdown-toggle#logout:hover,
span#login_widget > .button:active:focus,
#logout:active:focus,
span#login_widget > .button.active:focus,
#logout.active:focus,
.open > .dropdown-togglespan#login_widget > .button:focus,
.open > .dropdown-toggle#logout:focus,
span#login_widget > .button:active.focus,
#logout:active.focus,
span#login_widget > .button.active.focus,
#logout.active.focus,
.open > .dropdown-togglespan#login_widget > .button.focus,
.open > .dropdown-toggle#logout.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
span#login_widget > .button:active,
#logout:active,
span#login_widget > .button.active,
#logout.active,
.open > .dropdown-togglespan#login_widget > .button,
.open > .dropdown-toggle#logout {
  background-image: none;
}
span#login_widget > .button.disabled:hover,
#logout.disabled:hover,
span#login_widget > .button[disabled]:hover,
#logout[disabled]:hover,
fieldset[disabled] span#login_widget > .button:hover,
fieldset[disabled] #logout:hover,
span#login_widget > .button.disabled:focus,
#logout.disabled:focus,
span#login_widget > .button[disabled]:focus,
#logout[disabled]:focus,
fieldset[disabled] span#login_widget > .button:focus,
fieldset[disabled] #logout:focus,
span#login_widget > .button.disabled.focus,
#logout.disabled.focus,
span#login_widget > .button[disabled].focus,
#logout[disabled].focus,
fieldset[disabled] span#login_widget > .button.focus,
fieldset[disabled] #logout.focus {
  background-color: #fff;
  border-color: #ccc;
}
span#login_widget > .button .badge,
#logout .badge {
  color: #fff;
  background-color: #333;
}
.nav-header {
  text-transform: none;
}
#header > span {
  margin-top: 10px;
}
.modal_stretch .modal-dialog {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  min-height: 80vh;
}
.modal_stretch .modal-dialog .modal-body {
  max-height: calc(100vh - 200px);
  overflow: auto;
  flex: 1;
}
.modal-header {
  cursor: move;
}
@media (min-width: 768px) {
  .modal .modal-dialog {
    width: 700px;
  }
}
@media (min-width: 768px) {
  select.form-control {
    margin-left: 12px;
    margin-right: 12px;
  }
}
/*!
*
* IPython auth
*
*/
.center-nav {
  display: inline-block;
  margin-bottom: -4px;
}
[dir="rtl"] .center-nav form.pull-left {
  float: right !important;
  float: right;
}
[dir="rtl"] .center-nav .navbar-text {
  float: right;
}
[dir="rtl"] .navbar-inner {
  text-align: right;
}
[dir="rtl"] div.text-left {
  text-align: right;
}
/*!
*
* IPython tree view
*
*/
/* We need an invisible input field on top of the sentense*/
/* "Drag file onto the list ..." */
.alternate_upload {
  background-color: none;
  display: inline;
}
.alternate_upload.form {
  padding: 0;
  margin: 0;
}
.alternate_upload input.fileinput {
  position: absolute;
  display: block;
  width: 100%;
  height: 100%;
  overflow: hidden;
  cursor: pointer;
  opacity: 0;
  z-index: 2;
}
.alternate_upload .btn-xs > input.fileinput {
  margin: -1px -5px;
}
.alternate_upload .btn-upload {
  position: relative;
  height: 22px;
}
::-webkit-file-upload-button {
  cursor: pointer;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
ul#tabs {
  margin-bottom: 4px;
}
ul#tabs a {
  padding-top: 6px;
  padding-bottom: 4px;
}
[dir="rtl"] ul#tabs.nav-tabs > li {
  float: right;
}
[dir="rtl"] ul#tabs.nav.nav-tabs {
  padding-right: 0;
}
ul.breadcrumb a:focus,
ul.breadcrumb a:hover {
  text-decoration: none;
}
ul.breadcrumb i.icon-home {
  font-size: 16px;
  margin-right: 4px;
}
ul.breadcrumb span {
  color: #5e5e5e;
}
.list_toolbar {
  padding: 4px 0 4px 0;
  vertical-align: middle;
}
.list_toolbar .tree-buttons {
  padding-top: 1px;
}
[dir="rtl"] .list_toolbar .tree-buttons .pull-right {
  float: left !important;
  float: left;
}
[dir="rtl"] .list_toolbar .col-sm-4,
[dir="rtl"] .list_toolbar .col-sm-8 {
  float: right;
}
.dynamic-buttons {
  padding-top: 3px;
  display: inline-block;
}
.list_toolbar [class*="span"] {
  min-height: 24px;
}
.list_header {
  font-weight: bold;
  background-color: #EEE;
}
.list_placeholder {
  font-weight: bold;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
}
.list_container {
  margin-top: 4px;
  margin-bottom: 20px;
  border: 1px solid #ddd;
  border-radius: 2px;
}
.list_container > div {
  border-bottom: 1px solid #ddd;
}
.list_container > div:hover .list-item {
  background-color: red;
}
.list_container > div:last-child {
  border: none;
}
.list_item:hover .list_item {
  background-color: #ddd;
}
.list_item a {
  text-decoration: none;
}
.list_item:hover {
  background-color: #fafafa;
}
.list_header > div,
.list_item > div {
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
  line-height: 22px;
}
.list_header > div input,
.list_item > div input {
  margin-right: 7px;
  margin-left: 14px;
  vertical-align: text-bottom;
  line-height: 22px;
  position: relative;
  top: -1px;
}
.list_header > div .item_link,
.list_item > div .item_link {
  margin-left: -1px;
  vertical-align: baseline;
  line-height: 22px;
}
[dir="rtl"] .list_item > div input {
  margin-right: 0;
}
.new-file input[type=checkbox] {
  visibility: hidden;
}
.item_name {
  line-height: 22px;
  height: 24px;
}
.item_icon {
  font-size: 14px;
  color: #5e5e5e;
  margin-right: 7px;
  margin-left: 7px;
  line-height: 22px;
  vertical-align: baseline;
}
.item_modified {
  margin-right: 7px;
  margin-left: 7px;
}
[dir="rtl"] .item_modified.pull-right {
  float: left !important;
  float: left;
}
.item_buttons {
  line-height: 1em;
  margin-left: -5px;
}
.item_buttons .btn,
.item_buttons .btn-group,
.item_buttons .input-group {
  float: left;
}
.item_buttons > .btn,
.item_buttons > .btn-group,
.item_buttons > .input-group {
  margin-left: 5px;
}
.item_buttons .btn {
  min-width: 13ex;
}
.item_buttons .running-indicator {
  padding-top: 4px;
  color: #5cb85c;
}
.item_buttons .kernel-name {
  padding-top: 4px;
  color: #5bc0de;
  margin-right: 7px;
  float: left;
}
[dir="rtl"] .item_buttons.pull-right {
  float: left !important;
  float: left;
}
[dir="rtl"] .item_buttons .kernel-name {
  margin-left: 7px;
  float: right;
}
.toolbar_info {
  height: 24px;
  line-height: 24px;
}
.list_item input:not([type=checkbox]) {
  padding-top: 3px;
  padding-bottom: 3px;
  height: 22px;
  line-height: 14px;
  margin: 0px;
}
.highlight_text {
  color: blue;
}
#project_name {
  display: inline-block;
  padding-left: 7px;
  margin-left: -2px;
}
#project_name > .breadcrumb {
  padding: 0px;
  margin-bottom: 0px;
  background-color: transparent;
  font-weight: bold;
}
.sort_button {
  display: inline-block;
  padding-left: 7px;
}
[dir="rtl"] .sort_button.pull-right {
  float: left !important;
  float: left;
}
#tree-selector {
  padding-right: 0px;
}
#button-select-all {
  min-width: 50px;
}
[dir="rtl"] #button-select-all.btn {
  float: right ;
}
#select-all {
  margin-left: 7px;
  margin-right: 2px;
  margin-top: 2px;
  height: 16px;
}
[dir="rtl"] #select-all.pull-left {
  float: right !important;
  float: right;
}
.menu_icon {
  margin-right: 2px;
}
.tab-content .row {
  margin-left: 0px;
  margin-right: 0px;
}
.folder_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f114";
}
.folder_icon:before.fa-pull-left {
  margin-right: .3em;
}
.folder_icon:before.fa-pull-right {
  margin-left: .3em;
}
.folder_icon:before.pull-left {
  margin-right: .3em;
}
.folder_icon:before.pull-right {
  margin-left: .3em;
}
.notebook_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f02d";
  position: relative;
  top: -1px;
}
.notebook_icon:before.fa-pull-left {
  margin-right: .3em;
}
.notebook_icon:before.fa-pull-right {
  margin-left: .3em;
}
.notebook_icon:before.pull-left {
  margin-right: .3em;
}
.notebook_icon:before.pull-right {
  margin-left: .3em;
}
.running_notebook_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f02d";
  position: relative;
  top: -1px;
  color: #5cb85c;
}
.running_notebook_icon:before.fa-pull-left {
  margin-right: .3em;
}
.running_notebook_icon:before.fa-pull-right {
  margin-left: .3em;
}
.running_notebook_icon:before.pull-left {
  margin-right: .3em;
}
.running_notebook_icon:before.pull-right {
  margin-left: .3em;
}
.file_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f016";
  position: relative;
  top: -2px;
}
.file_icon:before.fa-pull-left {
  margin-right: .3em;
}
.file_icon:before.fa-pull-right {
  margin-left: .3em;
}
.file_icon:before.pull-left {
  margin-right: .3em;
}
.file_icon:before.pull-right {
  margin-left: .3em;
}
#notebook_toolbar .pull-right {
  padding-top: 0px;
  margin-right: -1px;
}
ul#new-menu {
  left: auto;
  right: 0;
}
#new-menu .dropdown-header {
  font-size: 10px;
  border-bottom: 1px solid #e5e5e5;
  padding: 0 0 3px;
  margin: -3px 20px 0;
}
.kernel-menu-icon {
  padding-right: 12px;
  width: 24px;
  content: "\f096";
}
.kernel-menu-icon:before {
  content: "\f096";
}
.kernel-menu-icon-current:before {
  content: "\f00c";
}
#tab_content {
  padding-top: 20px;
}
#running .panel-group .panel {
  margin-top: 3px;
  margin-bottom: 1em;
}
#running .panel-group .panel .panel-heading {
  background-color: #EEE;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
  line-height: 22px;
}
#running .panel-group .panel .panel-heading a:focus,
#running .panel-group .panel .panel-heading a:hover {
  text-decoration: none;
}
#running .panel-group .panel .panel-body {
  padding: 0px;
}
#running .panel-group .panel .panel-body .list_container {
  margin-top: 0px;
  margin-bottom: 0px;
  border: 0px;
  border-radius: 0px;
}
#running .panel-group .panel .panel-body .list_container .list_item {
  border-bottom: 1px solid #ddd;
}
#running .panel-group .panel .panel-body .list_container .list_item:last-child {
  border-bottom: 0px;
}
.delete-button {
  display: none;
}
.duplicate-button {
  display: none;
}
.rename-button {
  display: none;
}
.move-button {
  display: none;
}
.download-button {
  display: none;
}
.shutdown-button {
  display: none;
}
.dynamic-instructions {
  display: inline-block;
  padding-top: 4px;
}
/*!
*
* IPython text editor webapp
*
*/
.selected-keymap i.fa {
  padding: 0px 5px;
}
.selected-keymap i.fa:before {
  content: "\f00c";
}
#mode-menu {
  overflow: auto;
  max-height: 20em;
}
.edit_app #header {
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
.edit_app #menubar .navbar {
  /* Use a negative 1 bottom margin, so the border overlaps the border of the
    header */
  margin-bottom: -1px;
}
.dirty-indicator {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator.fa-pull-left {
  margin-right: .3em;
}
.dirty-indicator.fa-pull-right {
  margin-left: .3em;
}
.dirty-indicator.pull-left {
  margin-right: .3em;
}
.dirty-indicator.pull-right {
  margin-left: .3em;
}
.dirty-indicator-dirty {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator-dirty.fa-pull-left {
  margin-right: .3em;
}
.dirty-indicator-dirty.fa-pull-right {
  margin-left: .3em;
}
.dirty-indicator-dirty.pull-left {
  margin-right: .3em;
}
.dirty-indicator-dirty.pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator-clean.fa-pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean.fa-pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean.pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean.pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f00c";
}
.dirty-indicator-clean:before.fa-pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean:before.fa-pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean:before.pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean:before.pull-right {
  margin-left: .3em;
}
#filename {
  font-size: 16pt;
  display: table;
  padding: 0px 5px;
}
#current-mode {
  padding-left: 5px;
  padding-right: 5px;
}
#texteditor-backdrop {
  padding-top: 20px;
  padding-bottom: 20px;
}
@media not print {
  #texteditor-backdrop {
    background-color: #EEE;
  }
}
@media print {
  #texteditor-backdrop #texteditor-container .CodeMirror-gutter,
  #texteditor-backdrop #texteditor-container .CodeMirror-gutters {
    background-color: #fff;
  }
}
@media not print {
  #texteditor-backdrop #texteditor-container .CodeMirror-gutter,
  #texteditor-backdrop #texteditor-container .CodeMirror-gutters {
    background-color: #fff;
  }
}
@media not print {
  #texteditor-backdrop #texteditor-container {
    padding: 0px;
    background-color: #fff;
    -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
    box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  }
}
.CodeMirror-dialog {
  background-color: #fff;
}
/*!
*
* IPython notebook
*
*/
/* CSS font colors for translated ANSI escape sequences */
/* The color values are a mix of
   http://www.xcolors.net/dl/baskerville-ivorylight and
   http://www.xcolors.net/dl/euphrasia */
.ansi-black-fg {
  color: #3E424D;
}
.ansi-black-bg {
  background-color: #3E424D;
}
.ansi-black-intense-fg {
  color: #282C36;
}
.ansi-black-intense-bg {
  background-color: #282C36;
}
.ansi-red-fg {
  color: #E75C58;
}
.ansi-red-bg {
  background-color: #E75C58;
}
.ansi-red-intense-fg {
  color: #B22B31;
}
.ansi-red-intense-bg {
  background-color: #B22B31;
}
.ansi-green-fg {
  color: #00A250;
}
.ansi-green-bg {
  background-color: #00A250;
}
.ansi-green-intense-fg {
  color: #007427;
}
.ansi-green-intense-bg {
  background-color: #007427;
}
.ansi-yellow-fg {
  color: #DDB62B;
}
.ansi-yellow-bg {
  background-color: #DDB62B;
}
.ansi-yellow-intense-fg {
  color: #B27D12;
}
.ansi-yellow-intense-bg {
  background-color: #B27D12;
}
.ansi-blue-fg {
  color: #208FFB;
}
.ansi-blue-bg {
  background-color: #208FFB;
}
.ansi-blue-intense-fg {
  color: #0065CA;
}
.ansi-blue-intense-bg {
  background-color: #0065CA;
}
.ansi-magenta-fg {
  color: #D160C4;
}
.ansi-magenta-bg {
  background-color: #D160C4;
}
.ansi-magenta-intense-fg {
  color: #A03196;
}
.ansi-magenta-intense-bg {
  background-color: #A03196;
}
.ansi-cyan-fg {
  color: #60C6C8;
}
.ansi-cyan-bg {
  background-color: #60C6C8;
}
.ansi-cyan-intense-fg {
  color: #258F8F;
}
.ansi-cyan-intense-bg {
  background-color: #258F8F;
}
.ansi-white-fg {
  color: #C5C1B4;
}
.ansi-white-bg {
  background-color: #C5C1B4;
}
.ansi-white-intense-fg {
  color: #A1A6B2;
}
.ansi-white-intense-bg {
  background-color: #A1A6B2;
}
.ansi-default-inverse-fg {
  color: #FFFFFF;
}
.ansi-default-inverse-bg {
  background-color: #000000;
}
.ansi-bold {
  font-weight: bold;
}
.ansi-underline {
  text-decoration: underline;
}
/* The following styles are deprecated an will be removed in a future version */
.ansibold {
  font-weight: bold;
}
.ansi-inverse {
  outline: 0.5px dotted;
}
/* use dark versions for foreground, to improve visibility */
.ansiblack {
  color: black;
}
.ansired {
  color: darkred;
}
.ansigreen {
  color: darkgreen;
}
.ansiyellow {
  color: #c4a000;
}
.ansiblue {
  color: darkblue;
}
.ansipurple {
  color: darkviolet;
}
.ansicyan {
  color: steelblue;
}
.ansigray {
  color: gray;
}
/* and light for background, for the same reason */
.ansibgblack {
  background-color: black;
}
.ansibgred {
  background-color: red;
}
.ansibggreen {
  background-color: green;
}
.ansibgyellow {
  background-color: yellow;
}
.ansibgblue {
  background-color: blue;
}
.ansibgpurple {
  background-color: magenta;
}
.ansibgcyan {
  background-color: cyan;
}
.ansibggray {
  background-color: gray;
}
div.cell {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  border-radius: 2px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  border-width: 1px;
  border-style: solid;
  border-color: transparent;
  width: 100%;
  padding: 5px;
  /* This acts as a spacer between cells, that is outside the border */
  margin: 0px;
  outline: none;
  position: relative;
  overflow: visible;
}
div.cell:before {
  position: absolute;
  display: block;
  top: -1px;
  left: -1px;
  width: 5px;
  height: calc(100% +  2px);
  content: '';
  background: transparent;
}
div.cell.jupyter-soft-selected {
  border-left-color: #E3F2FD;
  border-left-width: 1px;
  padding-left: 5px;
  border-right-color: #E3F2FD;
  border-right-width: 1px;
  background: #E3F2FD;
}
@media print {
  div.cell.jupyter-soft-selected {
    border-color: transparent;
  }
}
div.cell.selected,
div.cell.selected.jupyter-soft-selected {
  border-color: #ababab;
}
div.cell.selected:before,
div.cell.selected.jupyter-soft-selected:before {
  position: absolute;
  display: block;
  top: -1px;
  left: -1px;
  width: 5px;
  height: calc(100% +  2px);
  content: '';
  background: #42A5F5;
}
@media print {
  div.cell.selected,
  div.cell.selected.jupyter-soft-selected {
    border-color: transparent;
  }
}
.edit_mode div.cell.selected {
  border-color: #66BB6A;
}
.edit_mode div.cell.selected:before {
  position: absolute;
  display: block;
  top: -1px;
  left: -1px;
  width: 5px;
  height: calc(100% +  2px);
  content: '';
  background: #66BB6A;
}
@media print {
  .edit_mode div.cell.selected {
    border-color: transparent;
  }
}
.prompt {
  /* This needs to be wide enough for 3 digit prompt numbers: In[100]: */
  min-width: 14ex;
  /* This padding is tuned to match the padding on the CodeMirror editor. */
  padding: 0.4em;
  margin: 0px;
  font-family: monospace;
  text-align: right;
  /* This has to match that of the the CodeMirror class line-height below */
  line-height: 1.21429em;
  /* Don't highlight prompt number selection */
  -webkit-touch-callout: none;
  -webkit-user-select: none;
  -khtml-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
  /* Use default cursor */
  cursor: default;
}
@media (max-width: 540px) {
  .prompt {
    text-align: left;
  }
}
div.inner_cell {
  min-width: 0;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
/* input_area and input_prompt must match in top border and margin for alignment */
div.input_area {
  border: 1px solid #cfcfcf;
  border-radius: 2px;
  background: #f7f7f7;
  line-height: 1.21429em;
}
/* This is needed so that empty prompt areas can collapse to zero height when there
   is no content in the output_subarea and the prompt. The main purpose of this is
   to make sure that empty JavaScript output_subareas have no height. */
div.prompt:empty {
  padding-top: 0;
  padding-bottom: 0;
}
div.unrecognized_cell {
  padding: 5px 5px 5px 0px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
div.unrecognized_cell .inner_cell {
  border-radius: 2px;
  padding: 5px;
  font-weight: bold;
  color: red;
  border: 1px solid #cfcfcf;
  background: #eaeaea;
}
div.unrecognized_cell .inner_cell a {
  color: inherit;
  text-decoration: none;
}
div.unrecognized_cell .inner_cell a:hover {
  color: inherit;
  text-decoration: none;
}
@media (max-width: 540px) {
  div.unrecognized_cell > div.prompt {
    display: none;
  }
}
div.code_cell {
  /* avoid page breaking on code cells when printing */
}
@media print {
  div.code_cell {
    page-break-inside: avoid;
  }
}
/* any special styling for code cells that are currently running goes here */
div.input {
  page-break-inside: avoid;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.input {
    /* Old browsers */
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-box-align: stretch;
    display: -moz-box;
    -moz-box-orient: vertical;
    -moz-box-align: stretch;
    display: box;
    box-orient: vertical;
    box-align: stretch;
    /* Modern browsers */
    display: flex;
    flex-direction: column;
    align-items: stretch;
  }
}
/* input_area and input_prompt must match in top border and margin for alignment */
div.input_prompt {
  color: #303F9F;
  border-top: 1px solid transparent;
}
div.input_area > div.highlight {
  margin: 0.4em;
  border: none;
  padding: 0px;
  background-color: transparent;
}
div.input_area > div.highlight > pre {
  margin: 0px;
  border: none;
  padding: 0px;
  background-color: transparent;
}
/* The following gets added to the <head> if it is detected that the user has a
 * monospace font with inconsistent normal/bold/italic height.  See
 * notebookmain.js.  Such fonts will have keywords vertically offset with
 * respect to the rest of the text.  The user should select a better font.
 * See: https://github.com/ipython/ipython/issues/1503
 *
 * .CodeMirror span {
 *      vertical-align: bottom;
 * }
 */
.CodeMirror {
  line-height: 1.21429em;
  /* Changed from 1em to our global default */
  font-size: 14px;
  height: auto;
  /* Changed to auto to autogrow */
  background: none;
  /* Changed from white to allow our bg to show through */
}
.CodeMirror-scroll {
  /*  The CodeMirror docs are a bit fuzzy on if overflow-y should be hidden or visible.*/
  /*  We have found that if it is visible, vertical scrollbars appear with font size changes.*/
  overflow-y: hidden;
  overflow-x: auto;
}
.CodeMirror-lines {
  /* In CM2, this used to be 0.4em, but in CM3 it went to 4px. We need the em value because */
  /* we have set a different line-height and want this to scale with that. */
  /* Note that this should set vertical padding only, since CodeMirror assumes
       that horizontal padding will be set on CodeMirror pre */
  padding: 0.4em 0;
}
.CodeMirror-linenumber {
  padding: 0 8px 0 4px;
}
.CodeMirror-gutters {
  border-bottom-left-radius: 2px;
  border-top-left-radius: 2px;
}
.CodeMirror pre {
  /* In CM3 this went to 4px from 0 in CM2. This sets horizontal padding only,
    use .CodeMirror-lines for vertical */
  padding: 0 0.4em;
  border: 0;
  border-radius: 0;
}
.CodeMirror-cursor {
  border-left: 1.4px solid black;
}
@media screen and (min-width: 2138px) and (max-width: 4319px) {
  .CodeMirror-cursor {
    border-left: 2px solid black;
  }
}
@media screen and (min-width: 4320px) {
  .CodeMirror-cursor {
    border-left: 4px solid black;
  }
}
/*

Original style from softwaremaniacs.org (c) Ivan Sagalaev <Maniac@SoftwareManiacs.Org>
Adapted from GitHub theme

*/
.highlight-base {
  color: #000;
}
.highlight-variable {
  color: #000;
}
.highlight-variable-2 {
  color: #1a1a1a;
}
.highlight-variable-3 {
  color: #333333;
}
.highlight-string {
  color: #BA2121;
}
.highlight-comment {
  color: #408080;
  font-style: italic;
}
.highlight-number {
  color: #080;
}
.highlight-atom {
  color: #88F;
}
.highlight-keyword {
  color: #008000;
  font-weight: bold;
}
.highlight-builtin {
  color: #008000;
}
.highlight-error {
  color: #f00;
}
.highlight-operator {
  color: #AA22FF;
  font-weight: bold;
}
.highlight-meta {
  color: #AA22FF;
}
/* previously not defined, copying from default codemirror */
.highlight-def {
  color: #00f;
}
.highlight-string-2 {
  color: #f50;
}
.highlight-qualifier {
  color: #555;
}
.highlight-bracket {
  color: #997;
}
.highlight-tag {
  color: #170;
}
.highlight-attribute {
  color: #00c;
}
.highlight-header {
  color: blue;
}
.highlight-quote {
  color: #090;
}
.highlight-link {
  color: #00c;
}
/* apply the same style to codemirror */
.cm-s-ipython span.cm-keyword {
  color: #008000;
  font-weight: bold;
}
.cm-s-ipython span.cm-atom {
  color: #88F;
}
.cm-s-ipython span.cm-number {
  color: #080;
}
.cm-s-ipython span.cm-def {
  color: #00f;
}
.cm-s-ipython span.cm-variable {
  color: #000;
}
.cm-s-ipython span.cm-operator {
  color: #AA22FF;
  font-weight: bold;
}
.cm-s-ipython span.cm-variable-2 {
  color: #1a1a1a;
}
.cm-s-ipython span.cm-variable-3 {
  color: #333333;
}
.cm-s-ipython span.cm-comment {
  color: #408080;
  font-style: italic;
}
.cm-s-ipython span.cm-string {
  color: #BA2121;
}
.cm-s-ipython span.cm-string-2 {
  color: #f50;
}
.cm-s-ipython span.cm-meta {
  color: #AA22FF;
}
.cm-s-ipython span.cm-qualifier {
  color: #555;
}
.cm-s-ipython span.cm-builtin {
  color: #008000;
}
.cm-s-ipython span.cm-bracket {
  color: #997;
}
.cm-s-ipython span.cm-tag {
  color: #170;
}
.cm-s-ipython span.cm-attribute {
  color: #00c;
}
.cm-s-ipython span.cm-header {
  color: blue;
}
.cm-s-ipython span.cm-quote {
  color: #090;
}
.cm-s-ipython span.cm-link {
  color: #00c;
}
.cm-s-ipython span.cm-error {
  color: #f00;
}
.cm-s-ipython span.cm-tab {
  background: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAMCAYAAAAkuj5RAAAAAXNSR0IArs4c6QAAAGFJREFUSMft1LsRQFAQheHPowAKoACx3IgEKtaEHujDjORSgWTH/ZOdnZOcM/sgk/kFFWY0qV8foQwS4MKBCS3qR6ixBJvElOobYAtivseIE120FaowJPN75GMu8j/LfMwNjh4HUpwg4LUAAAAASUVORK5CYII=);
  background-position: right;
  background-repeat: no-repeat;
}
div.output_wrapper {
  /* this position must be relative to enable descendents to be absolute within it */
  position: relative;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  z-index: 1;
}
/* class for the output area when it should be height-limited */
div.output_scroll {
  /* ideally, this would be max-height, but FF barfs all over that */
  height: 24em;
  /* FF needs this *and the wrapper* to specify full width, or it will shrinkwrap */
  width: 100%;
  overflow: auto;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
  box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
  display: block;
}
/* output div while it is collapsed */
div.output_collapsed {
  margin: 0px;
  padding: 0px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
div.out_prompt_overlay {
  height: 100%;
  padding: 0px 0.4em;
  position: absolute;
  border-radius: 2px;
}
div.out_prompt_overlay:hover {
  /* use inner shadow to get border that is computed the same on WebKit/FF */
  -webkit-box-shadow: inset 0 0 1px #000;
  box-shadow: inset 0 0 1px #000;
  background: rgba(240, 240, 240, 0.5);
}
div.output_prompt {
  color: #D84315;
}
/* This class is the outer container of all output sections. */
div.output_area {
  padding: 0px;
  page-break-inside: avoid;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
div.output_area .MathJax_Display {
  text-align: left !important;
}
div.output_area .rendered_html table {
  margin-left: 0;
  margin-right: 0;
}
div.output_area .rendered_html img {
  margin-left: 0;
  margin-right: 0;
}
div.output_area img,
div.output_area svg {
  max-width: 100%;
  height: auto;
}
div.output_area img.unconfined,
div.output_area svg.unconfined {
  max-width: none;
}
div.output_area .mglyph > img {
  max-width: none;
}
/* This is needed to protect the pre formating from global settings such
   as that of bootstrap */
.output {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.output_area {
    /* Old browsers */
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-box-align: stretch;
    display: -moz-box;
    -moz-box-orient: vertical;
    -moz-box-align: stretch;
    display: box;
    box-orient: vertical;
    box-align: stretch;
    /* Modern browsers */
    display: flex;
    flex-direction: column;
    align-items: stretch;
  }
}
div.output_area pre {
  margin: 0;
  padding: 1px 0 1px 0;
  border: 0;
  vertical-align: baseline;
  color: black;
  background-color: transparent;
  border-radius: 0;
}
/* This class is for the output subarea inside the output_area and after
   the prompt div. */
div.output_subarea {
  overflow-x: auto;
  padding: 0.4em;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
  max-width: calc(100% - 14ex);
}
div.output_scroll div.output_subarea {
  overflow-x: visible;
}
/* The rest of the output_* classes are for special styling of the different
   output types */
/* all text output has this class: */
div.output_text {
  text-align: left;
  color: #000;
  /* This has to match that of the the CodeMirror class line-height below */
  line-height: 1.21429em;
}
/* stdout/stderr are 'text' as well as 'stream', but execute_result/error are *not* streams */
div.output_stderr {
  background: #fdd;
  /* very light red background for stderr */
}
div.output_latex {
  text-align: left;
}
/* Empty output_javascript divs should have no height */
div.output_javascript:empty {
  padding: 0;
}
.js-error {
  color: darkred;
}
/* raw_input styles */
div.raw_input_container {
  line-height: 1.21429em;
  padding-top: 5px;
}
pre.raw_input_prompt {
  /* nothing needed here. */
}
input.raw_input {
  font-family: monospace;
  font-size: inherit;
  color: inherit;
  width: auto;
  /* make sure input baseline aligns with prompt */
  vertical-align: baseline;
  /* padding + margin = 0.5em between prompt and cursor */
  padding: 0em 0.25em;
  margin: 0em 0.25em;
}
input.raw_input:focus {
  box-shadow: none;
}
p.p-space {
  margin-bottom: 10px;
}
div.output_unrecognized {
  padding: 5px;
  font-weight: bold;
  color: red;
}
div.output_unrecognized a {
  color: inherit;
  text-decoration: none;
}
div.output_unrecognized a:hover {
  color: inherit;
  text-decoration: none;
}
.rendered_html {
  color: #000;
  /* any extras will just be numbers: */
}
.rendered_html em {
  font-style: italic;
}
.rendered_html strong {
  font-weight: bold;
}
.rendered_html u {
  text-decoration: underline;
}
.rendered_html :link {
  text-decoration: underline;
}
.rendered_html :visited {
  text-decoration: underline;
}
.rendered_html h1 {
  font-size: 185.7%;
  margin: 1.08em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h2 {
  font-size: 157.1%;
  margin: 1.27em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h3 {
  font-size: 128.6%;
  margin: 1.55em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h4 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h5 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
  font-style: italic;
}
.rendered_html h6 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
  font-style: italic;
}
.rendered_html h1:first-child {
  margin-top: 0.538em;
}
.rendered_html h2:first-child {
  margin-top: 0.636em;
}
.rendered_html h3:first-child {
  margin-top: 0.777em;
}
.rendered_html h4:first-child {
  margin-top: 1em;
}
.rendered_html h5:first-child {
  margin-top: 1em;
}
.rendered_html h6:first-child {
  margin-top: 1em;
}
.rendered_html ul:not(.list-inline),
.rendered_html ol:not(.list-inline) {
  padding-left: 2em;
}
.rendered_html ul {
  list-style: disc;
}
.rendered_html ul ul {
  list-style: square;
  margin-top: 0;
}
.rendered_html ul ul ul {
  list-style: circle;
}
.rendered_html ol {
  list-style: decimal;
}
.rendered_html ol ol {
  list-style: upper-alpha;
  margin-top: 0;
}
.rendered_html ol ol ol {
  list-style: lower-alpha;
}
.rendered_html ol ol ol ol {
  list-style: lower-roman;
}
.rendered_html ol ol ol ol ol {
  list-style: decimal;
}
.rendered_html * + ul {
  margin-top: 1em;
}
.rendered_html * + ol {
  margin-top: 1em;
}
.rendered_html hr {
  color: black;
  background-color: black;
}
.rendered_html pre {
  margin: 1em 2em;
  padding: 0px;
  background-color: #fff;
}
.rendered_html code {
  background-color: #eff0f1;
}
.rendered_html p code {
  padding: 1px 5px;
}
.rendered_html pre code {
  background-color: #fff;
}
.rendered_html pre,
.rendered_html code {
  border: 0;
  color: #000;
  font-size: 100%;
}
.rendered_html blockquote {
  margin: 1em 2em;
}
.rendered_html table {
  margin-left: auto;
  margin-right: auto;
  border: none;
  border-collapse: collapse;
  border-spacing: 0;
  color: black;
  font-size: 12px;
  table-layout: fixed;
}
.rendered_html thead {
  border-bottom: 1px solid black;
  vertical-align: bottom;
}
.rendered_html tr,
.rendered_html th,
.rendered_html td {
  text-align: right;
  vertical-align: middle;
  padding: 0.5em 0.5em;
  line-height: normal;
  white-space: normal;
  max-width: none;
  border: none;
}
.rendered_html th {
  font-weight: bold;
}
.rendered_html tbody tr:nth-child(odd) {
  background: #f5f5f5;
}
.rendered_html tbody tr:hover {
  background: rgba(66, 165, 245, 0.2);
}
.rendered_html * + table {
  margin-top: 1em;
}
.rendered_html p {
  text-align: left;
}
.rendered_html * + p {
  margin-top: 1em;
}
.rendered_html img {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
.rendered_html * + img {
  margin-top: 1em;
}
.rendered_html img,
.rendered_html svg {
  max-width: 100%;
  height: auto;
}
.rendered_html img.unconfined,
.rendered_html svg.unconfined {
  max-width: none;
}
.rendered_html .alert {
  margin-bottom: initial;
}
.rendered_html * + .alert {
  margin-top: 1em;
}
[dir="rtl"] .rendered_html p {
  text-align: right;
}
div.text_cell {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.text_cell > div.prompt {
    display: none;
  }
}
div.text_cell_render {
  /*font-family: "Helvetica Neue", Arial, Helvetica, Geneva, sans-serif;*/
  outline: none;
  resize: none;
  width: inherit;
  border-style: none;
  padding: 0.5em 0.5em 0.5em 0.4em;
  color: #000;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
a.anchor-link:link {
  text-decoration: none;
  padding: 0px 20px;
  visibility: hidden;
}
h1:hover .anchor-link,
h2:hover .anchor-link,
h3:hover .anchor-link,
h4:hover .anchor-link,
h5:hover .anchor-link,
h6:hover .anchor-link {
  visibility: visible;
}
.text_cell.rendered .input_area {
  display: none;
}
.text_cell.rendered .rendered_html {
  overflow-x: auto;
  overflow-y: hidden;
}
.text_cell.rendered .rendered_html tr,
.text_cell.rendered .rendered_html th,
.text_cell.rendered .rendered_html td {
  max-width: none;
}
.text_cell.unrendered .text_cell_render {
  display: none;
}
.text_cell .dropzone .input_area {
  border: 2px dashed #bababa;
  margin: -1px;
}
.cm-header-1,
.cm-header-2,
.cm-header-3,
.cm-header-4,
.cm-header-5,
.cm-header-6 {
  font-weight: bold;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}
.cm-header-1 {
  font-size: 185.7%;
}
.cm-header-2 {
  font-size: 157.1%;
}
.cm-header-3 {
  font-size: 128.6%;
}
.cm-header-4 {
  font-size: 110%;
}
.cm-header-5 {
  font-size: 100%;
  font-style: italic;
}
.cm-header-6 {
  font-size: 100%;
  font-style: italic;
}
/*!
*
* IPython notebook webapp
*
*/
@media (max-width: 767px) {
  .notebook_app {
    padding-left: 0px;
    padding-right: 0px;
  }
}
#ipython-main-app {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  height: 100%;
}
div#notebook_panel {
  margin: 0px;
  padding: 0px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  height: 100%;
}
div#notebook {
  font-size: 14px;
  line-height: 20px;
  overflow-y: hidden;
  overflow-x: auto;
  width: 100%;
  /* This spaces the page away from the edge of the notebook area */
  padding-top: 20px;
  margin: 0px;
  outline: none;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  min-height: 100%;
}
@media not print {
  #notebook-container {
    padding: 15px;
    background-color: #fff;
    min-height: 0;
    -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
    box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  }
}
@media print {
  #notebook-container {
    width: 100%;
  }
}
div.ui-widget-content {
  border: 1px solid #ababab;
  outline: none;
}
pre.dialog {
  background-color: #f7f7f7;
  border: 1px solid #ddd;
  border-radius: 2px;
  padding: 0.4em;
  padding-left: 2em;
}
p.dialog {
  padding: 0.2em;
}
/* Word-wrap output correctly.  This is the CSS3 spelling, though Firefox seems
   to not honor it correctly.  Webkit browsers (Chrome, rekonq, Safari) do.
 */
pre,
code,
kbd,
samp {
  white-space: pre-wrap;
}
#fonttest {
  font-family: monospace;
}
p {
  margin-bottom: 0;
}
.end_space {
  min-height: 100px;
  transition: height .2s ease;
}
.notebook_app > #header {
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
@media not print {
  .notebook_app {
    background-color: #EEE;
  }
}
kbd {
  border-style: solid;
  border-width: 1px;
  box-shadow: none;
  margin: 2px;
  padding-left: 2px;
  padding-right: 2px;
  padding-top: 1px;
  padding-bottom: 1px;
}
.jupyter-keybindings {
  padding: 1px;
  line-height: 24px;
  border-bottom: 1px solid gray;
}
.jupyter-keybindings input {
  margin: 0;
  padding: 0;
  border: none;
}
.jupyter-keybindings i {
  padding: 6px;
}
.well code {
  background-color: #ffffff;
  border-color: #ababab;
  border-width: 1px;
  border-style: solid;
  padding: 2px;
  padding-top: 1px;
  padding-bottom: 1px;
}
/* CSS for the cell toolbar */
.celltoolbar {
  border: thin solid #CFCFCF;
  border-bottom: none;
  background: #EEE;
  border-radius: 2px 2px 0px 0px;
  width: 100%;
  height: 29px;
  padding-right: 4px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
  /* Old browsers */
  -webkit-box-pack: end;
  -moz-box-pack: end;
  box-pack: end;
  /* Modern browsers */
  justify-content: flex-end;
  display: -webkit-flex;
}
@media print {
  .celltoolbar {
    display: none;
  }
}
.ctb_hideshow {
  display: none;
  vertical-align: bottom;
}
/* ctb_show is added to the ctb_hideshow div to show the cell toolbar.
   Cell toolbars are only shown when the ctb_global_show class is also set.
*/
.ctb_global_show .ctb_show.ctb_hideshow {
  display: block;
}
.ctb_global_show .ctb_show + .input_area,
.ctb_global_show .ctb_show + div.text_cell_input,
.ctb_global_show .ctb_show ~ div.text_cell_render {
  border-top-right-radius: 0px;
  border-top-left-radius: 0px;
}
.ctb_global_show .ctb_show ~ div.text_cell_render {
  border: 1px solid #cfcfcf;
}
.celltoolbar {
  font-size: 87%;
  padding-top: 3px;
}
.celltoolbar select {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
  width: inherit;
  font-size: inherit;
  height: 22px;
  padding: 0px;
  display: inline-block;
}
.celltoolbar select:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.celltoolbar select::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.celltoolbar select:-ms-input-placeholder {
  color: #999;
}
.celltoolbar select::-webkit-input-placeholder {
  color: #999;
}
.celltoolbar select::-ms-expand {
  border: 0;
  background-color: transparent;
}
.celltoolbar select[disabled],
.celltoolbar select[readonly],
fieldset[disabled] .celltoolbar select {
  background-color: #eeeeee;
  opacity: 1;
}
.celltoolbar select[disabled],
fieldset[disabled] .celltoolbar select {
  cursor: not-allowed;
}
textarea.celltoolbar select {
  height: auto;
}
select.celltoolbar select {
  height: 30px;
  line-height: 30px;
}
textarea.celltoolbar select,
select[multiple].celltoolbar select {
  height: auto;
}
.celltoolbar label {
  margin-left: 5px;
  margin-right: 5px;
}
.tags_button_container {
  width: 100%;
  display: flex;
}
.tag-container {
  display: flex;
  flex-direction: row;
  flex-grow: 1;
  overflow: hidden;
  position: relative;
}
.tag-container > * {
  margin: 0 4px;
}
.remove-tag-btn {
  margin-left: 4px;
}
.tags-input {
  display: flex;
}
.cell-tag:last-child:after {
  content: "";
  position: absolute;
  right: 0;
  width: 40px;
  height: 100%;
  /* Fade to background color of cell toolbar */
  background: linear-gradient(to right, rgba(0, 0, 0, 0), #EEE);
}
.tags-input > * {
  margin-left: 4px;
}
.cell-tag,
.tags-input input,
.tags-input button {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
  box-shadow: none;
  width: inherit;
  font-size: inherit;
  height: 22px;
  line-height: 22px;
  padding: 0px 4px;
  display: inline-block;
}
.cell-tag:focus,
.tags-input input:focus,
.tags-input button:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.cell-tag::-moz-placeholder,
.tags-input input::-moz-placeholder,
.tags-input button::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.cell-tag:-ms-input-placeholder,
.tags-input input:-ms-input-placeholder,
.tags-input button:-ms-input-placeholder {
  color: #999;
}
.cell-tag::-webkit-input-placeholder,
.tags-input input::-webkit-input-placeholder,
.tags-input button::-webkit-input-placeholder {
  color: #999;
}
.cell-tag::-ms-expand,
.tags-input input::-ms-expand,
.tags-input button::-ms-expand {
  border: 0;
  background-color: transparent;
}
.cell-tag[disabled],
.tags-input input[disabled],
.tags-input button[disabled],
.cell-tag[readonly],
.tags-input input[readonly],
.tags-input button[readonly],
fieldset[disabled] .cell-tag,
fieldset[disabled] .tags-input input,
fieldset[disabled] .tags-input button {
  background-color: #eeeeee;
  opacity: 1;
}
.cell-tag[disabled],
.tags-input input[disabled],
.tags-input button[disabled],
fieldset[disabled] .cell-tag,
fieldset[disabled] .tags-input input,
fieldset[disabled] .tags-input button {
  cursor: not-allowed;
}
textarea.cell-tag,
textarea.tags-input input,
textarea.tags-input button {
  height: auto;
}
select.cell-tag,
select.tags-input input,
select.tags-input button {
  height: 30px;
  line-height: 30px;
}
textarea.cell-tag,
textarea.tags-input input,
textarea.tags-input button,
select[multiple].cell-tag,
select[multiple].tags-input input,
select[multiple].tags-input button {
  height: auto;
}
.cell-tag,
.tags-input button {
  padding: 0px 4px;
}
.cell-tag {
  background-color: #fff;
  white-space: nowrap;
}
.tags-input input[type=text]:focus {
  outline: none;
  box-shadow: none;
  border-color: #ccc;
}
.completions {
  position: absolute;
  z-index: 110;
  overflow: hidden;
  border: 1px solid #ababab;
  border-radius: 2px;
  -webkit-box-shadow: 0px 6px 10px -1px #adadad;
  box-shadow: 0px 6px 10px -1px #adadad;
  line-height: 1;
}
.completions select {
  background: white;
  outline: none;
  border: none;
  padding: 0px;
  margin: 0px;
  overflow: auto;
  font-family: monospace;
  font-size: 110%;
  color: #000;
  width: auto;
}
.completions select option.context {
  color: #286090;
}
#kernel_logo_widget .current_kernel_logo {
  display: none;
  margin-top: -1px;
  margin-bottom: -1px;
  width: 32px;
  height: 32px;
}
[dir="rtl"] #kernel_logo_widget {
  float: left !important;
  float: left;
}
.modal .modal-body .move-path {
  display: flex;
  flex-direction: row;
  justify-content: space;
  align-items: center;
}
.modal .modal-body .move-path .server-root {
  padding-right: 20px;
}
.modal .modal-body .move-path .path-input {
  flex: 1;
}
#menubar {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  margin-top: 1px;
}
#menubar .navbar {
  border-top: 1px;
  border-radius: 0px 0px 2px 2px;
  margin-bottom: 0px;
}
#menubar .navbar-toggle {
  float: left;
  padding-top: 7px;
  padding-bottom: 7px;
  border: none;
}
#menubar .navbar-collapse {
  clear: left;
}
[dir="rtl"] #menubar .navbar-toggle {
  float: right;
}
[dir="rtl"] #menubar .navbar-collapse {
  clear: right;
}
[dir="rtl"] #menubar .navbar-nav {
  float: right;
}
[dir="rtl"] #menubar .nav {
  padding-right: 0px;
}
[dir="rtl"] #menubar .navbar-nav > li {
  float: right;
}
[dir="rtl"] #menubar .navbar-right {
  float: left !important;
}
[dir="rtl"] ul.dropdown-menu {
  text-align: right;
  left: auto;
}
[dir="rtl"] ul#new-menu.dropdown-menu {
  right: auto;
  left: 0;
}
.nav-wrapper {
  border-bottom: 1px solid #e7e7e7;
}
i.menu-icon {
  padding-top: 4px;
}
[dir="rtl"] i.menu-icon.pull-right {
  float: left !important;
  float: left;
}
ul#help_menu li a {
  overflow: hidden;
  padding-right: 2.2em;
}
ul#help_menu li a i {
  margin-right: -1.2em;
}
[dir="rtl"] ul#help_menu li a {
  padding-left: 2.2em;
}
[dir="rtl"] ul#help_menu li a i {
  margin-right: 0;
  margin-left: -1.2em;
}
[dir="rtl"] ul#help_menu li a i.pull-right {
  float: left !important;
  float: left;
}
.dropdown-submenu {
  position: relative;
}
.dropdown-submenu > .dropdown-menu {
  top: 0;
  left: 100%;
  margin-top: -6px;
  margin-left: -1px;
}
[dir="rtl"] .dropdown-submenu > .dropdown-menu {
  right: 100%;
  margin-right: -1px;
}
.dropdown-submenu:hover > .dropdown-menu {
  display: block;
}
.dropdown-submenu > a:after {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  display: block;
  content: "\f0da";
  float: right;
  color: #333333;
  margin-top: 2px;
  margin-right: -10px;
}
.dropdown-submenu > a:after.fa-pull-left {
  margin-right: .3em;
}
.dropdown-submenu > a:after.fa-pull-right {
  margin-left: .3em;
}
.dropdown-submenu > a:after.pull-left {
  margin-right: .3em;
}
.dropdown-submenu > a:after.pull-right {
  margin-left: .3em;
}
[dir="rtl"] .dropdown-submenu > a:after {
  float: left;
  content: "\f0d9";
  margin-right: 0;
  margin-left: -10px;
}
.dropdown-submenu:hover > a:after {
  color: #262626;
}
.dropdown-submenu.pull-left {
  float: none;
}
.dropdown-submenu.pull-left > .dropdown-menu {
  left: -100%;
  margin-left: 10px;
}
#notification_area {
  float: right !important;
  float: right;
  z-index: 10;
}
[dir="rtl"] #notification_area {
  float: left !important;
  float: left;
}
.indicator_area {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
}
[dir="rtl"] .indicator_area {
  float: left !important;
  float: left;
}
#kernel_indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
  border-left: 1px solid;
}
#kernel_indicator .kernel_indicator_name {
  padding-left: 5px;
  padding-right: 5px;
}
[dir="rtl"] #kernel_indicator {
  float: left !important;
  float: left;
  border-left: 0;
  border-right: 1px solid;
}
#modal_indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
}
[dir="rtl"] #modal_indicator {
  float: left !important;
  float: left;
}
#readonly-indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
  margin-top: 2px;
  margin-bottom: 0px;
  margin-left: 0px;
  margin-right: 0px;
  display: none;
}
.modal_indicator:before {
  width: 1.28571429em;
  text-align: center;
}
.edit_mode .modal_indicator:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f040";
}
.edit_mode .modal_indicator:before.fa-pull-left {
  margin-right: .3em;
}
.edit_mode .modal_indicator:before.fa-pull-right {
  margin-left: .3em;
}
.edit_mode .modal_indicator:before.pull-left {
  margin-right: .3em;
}
.edit_mode .modal_indicator:before.pull-right {
  margin-left: .3em;
}
.command_mode .modal_indicator:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: ' ';
}
.command_mode .modal_indicator:before.fa-pull-left {
  margin-right: .3em;
}
.command_mode .modal_indicator:before.fa-pull-right {
  margin-left: .3em;
}
.command_mode .modal_indicator:before.pull-left {
  margin-right: .3em;
}
.command_mode .modal_indicator:before.pull-right {
  margin-left: .3em;
}
.kernel_idle_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f10c";
}
.kernel_idle_icon:before.fa-pull-left {
  margin-right: .3em;
}
.kernel_idle_icon:before.fa-pull-right {
  margin-left: .3em;
}
.kernel_idle_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_idle_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_busy_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f111";
}
.kernel_busy_icon:before.fa-pull-left {
  margin-right: .3em;
}
.kernel_busy_icon:before.fa-pull-right {
  margin-left: .3em;
}
.kernel_busy_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_busy_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_dead_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f1e2";
}
.kernel_dead_icon:before.fa-pull-left {
  margin-right: .3em;
}
.kernel_dead_icon:before.fa-pull-right {
  margin-left: .3em;
}
.kernel_dead_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_dead_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_disconnected_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f127";
}
.kernel_disconnected_icon:before.fa-pull-left {
  margin-right: .3em;
}
.kernel_disconnected_icon:before.fa-pull-right {
  margin-left: .3em;
}
.kernel_disconnected_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_disconnected_icon:before.pull-right {
  margin-left: .3em;
}
.notification_widget {
  color: #777;
  z-index: 10;
  background: rgba(240, 240, 240, 0.5);
  margin-right: 4px;
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
.notification_widget:focus,
.notification_widget.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
.notification_widget:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.notification_widget:active,
.notification_widget.active,
.open > .dropdown-toggle.notification_widget {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.notification_widget:active:hover,
.notification_widget.active:hover,
.open > .dropdown-toggle.notification_widget:hover,
.notification_widget:active:focus,
.notification_widget.active:focus,
.open > .dropdown-toggle.notification_widget:focus,
.notification_widget:active.focus,
.notification_widget.active.focus,
.open > .dropdown-toggle.notification_widget.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
.notification_widget:active,
.notification_widget.active,
.open > .dropdown-toggle.notification_widget {
  background-image: none;
}
.notification_widget.disabled:hover,
.notification_widget[disabled]:hover,
fieldset[disabled] .notification_widget:hover,
.notification_widget.disabled:focus,
.notification_widget[disabled]:focus,
fieldset[disabled] .notification_widget:focus,
.notification_widget.disabled.focus,
.notification_widget[disabled].focus,
fieldset[disabled] .notification_widget.focus {
  background-color: #fff;
  border-color: #ccc;
}
.notification_widget .badge {
  color: #fff;
  background-color: #333;
}
.notification_widget.warning {
  color: #fff;
  background-color: #f0ad4e;
  border-color: #eea236;
}
.notification_widget.warning:focus,
.notification_widget.warning.focus {
  color: #fff;
  background-color: #ec971f;
  border-color: #985f0d;
}
.notification_widget.warning:hover {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.notification_widget.warning:active,
.notification_widget.warning.active,
.open > .dropdown-toggle.notification_widget.warning {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.notification_widget.warning:active:hover,
.notification_widget.warning.active:hover,
.open > .dropdown-toggle.notification_widget.warning:hover,
.notification_widget.warning:active:focus,
.notification_widget.warning.active:focus,
.open > .dropdown-toggle.notification_widget.warning:focus,
.notification_widget.warning:active.focus,
.notification_widget.warning.active.focus,
.open > .dropdown-toggle.notification_widget.warning.focus {
  color: #fff;
  background-color: #d58512;
  border-color: #985f0d;
}
.notification_widget.warning:active,
.notification_widget.warning.active,
.open > .dropdown-toggle.notification_widget.warning {
  background-image: none;
}
.notification_widget.warning.disabled:hover,
.notification_widget.warning[disabled]:hover,
fieldset[disabled] .notification_widget.warning:hover,
.notification_widget.warning.disabled:focus,
.notification_widget.warning[disabled]:focus,
fieldset[disabled] .notification_widget.warning:focus,
.notification_widget.warning.disabled.focus,
.notification_widget.warning[disabled].focus,
fieldset[disabled] .notification_widget.warning.focus {
  background-color: #f0ad4e;
  border-color: #eea236;
}
.notification_widget.warning .badge {
  color: #f0ad4e;
  background-color: #fff;
}
.notification_widget.success {
  color: #fff;
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.notification_widget.success:focus,
.notification_widget.success.focus {
  color: #fff;
  background-color: #449d44;
  border-color: #255625;
}
.notification_widget.success:hover {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.notification_widget.success:active,
.notification_widget.success.active,
.open > .dropdown-toggle.notification_widget.success {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.notification_widget.success:active:hover,
.notification_widget.success.active:hover,
.open > .dropdown-toggle.notification_widget.success:hover,
.notification_widget.success:active:focus,
.notification_widget.success.active:focus,
.open > .dropdown-toggle.notification_widget.success:focus,
.notification_widget.success:active.focus,
.notification_widget.success.active.focus,
.open > .dropdown-toggle.notification_widget.success.focus {
  color: #fff;
  background-color: #398439;
  border-color: #255625;
}
.notification_widget.success:active,
.notification_widget.success.active,
.open > .dropdown-toggle.notification_widget.success {
  background-image: none;
}
.notification_widget.success.disabled:hover,
.notification_widget.success[disabled]:hover,
fieldset[disabled] .notification_widget.success:hover,
.notification_widget.success.disabled:focus,
.notification_widget.success[disabled]:focus,
fieldset[disabled] .notification_widget.success:focus,
.notification_widget.success.disabled.focus,
.notification_widget.success[disabled].focus,
fieldset[disabled] .notification_widget.success.focus {
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.notification_widget.success .badge {
  color: #5cb85c;
  background-color: #fff;
}
.notification_widget.info {
  color: #fff;
  background-color: #5bc0de;
  border-color: #46b8da;
}
.notification_widget.info:focus,
.notification_widget.info.focus {
  color: #fff;
  background-color: #31b0d5;
  border-color: #1b6d85;
}
.notification_widget.info:hover {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.notification_widget.info:active,
.notification_widget.info.active,
.open > .dropdown-toggle.notification_widget.info {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.notification_widget.info:active:hover,
.notification_widget.info.active:hover,
.open > .dropdown-toggle.notification_widget.info:hover,
.notification_widget.info:active:focus,
.notification_widget.info.active:focus,
.open > .dropdown-toggle.notification_widget.info:focus,
.notification_widget.info:active.focus,
.notification_widget.info.active.focus,
.open > .dropdown-toggle.notification_widget.info.focus {
  color: #fff;
  background-color: #269abc;
  border-color: #1b6d85;
}
.notification_widget.info:active,
.notification_widget.info.active,
.open > .dropdown-toggle.notification_widget.info {
  background-image: none;
}
.notification_widget.info.disabled:hover,
.notification_widget.info[disabled]:hover,
fieldset[disabled] .notification_widget.info:hover,
.notification_widget.info.disabled:focus,
.notification_widget.info[disabled]:focus,
fieldset[disabled] .notification_widget.info:focus,
.notification_widget.info.disabled.focus,
.notification_widget.info[disabled].focus,
fieldset[disabled] .notification_widget.info.focus {
  background-color: #5bc0de;
  border-color: #46b8da;
}
.notification_widget.info .badge {
  color: #5bc0de;
  background-color: #fff;
}
.notification_widget.danger {
  color: #fff;
  background-color: #d9534f;
  border-color: #d43f3a;
}
.notification_widget.danger:focus,
.notification_widget.danger.focus {
  color: #fff;
  background-color: #c9302c;
  border-color: #761c19;
}
.notification_widget.danger:hover {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.notification_widget.danger:active,
.notification_widget.danger.active,
.open > .dropdown-toggle.notification_widget.danger {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.notification_widget.danger:active:hover,
.notification_widget.danger.active:hover,
.open > .dropdown-toggle.notification_widget.danger:hover,
.notification_widget.danger:active:focus,
.notification_widget.danger.active:focus,
.open > .dropdown-toggle.notification_widget.danger:focus,
.notification_widget.danger:active.focus,
.notification_widget.danger.active.focus,
.open > .dropdown-toggle.notification_widget.danger.focus {
  color: #fff;
  background-color: #ac2925;
  border-color: #761c19;
}
.notification_widget.danger:active,
.notification_widget.danger.active,
.open > .dropdown-toggle.notification_widget.danger {
  background-image: none;
}
.notification_widget.danger.disabled:hover,
.notification_widget.danger[disabled]:hover,
fieldset[disabled] .notification_widget.danger:hover,
.notification_widget.danger.disabled:focus,
.notification_widget.danger[disabled]:focus,
fieldset[disabled] .notification_widget.danger:focus,
.notification_widget.danger.disabled.focus,
.notification_widget.danger[disabled].focus,
fieldset[disabled] .notification_widget.danger.focus {
  background-color: #d9534f;
  border-color: #d43f3a;
}
.notification_widget.danger .badge {
  color: #d9534f;
  background-color: #fff;
}
div#pager {
  background-color: #fff;
  font-size: 14px;
  line-height: 20px;
  overflow: hidden;
  display: none;
  position: fixed;
  bottom: 0px;
  width: 100%;
  max-height: 50%;
  padding-top: 8px;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  /* Display over codemirror */
  z-index: 100;
  /* Hack which prevents jquery ui resizable from changing top. */
  top: auto !important;
}
div#pager pre {
  line-height: 1.21429em;
  color: #000;
  background-color: #f7f7f7;
  padding: 0.4em;
}
div#pager #pager-button-area {
  position: absolute;
  top: 8px;
  right: 20px;
}
div#pager #pager-contents {
  position: relative;
  overflow: auto;
  width: 100%;
  height: 100%;
}
div#pager #pager-contents #pager-container {
  position: relative;
  padding: 15px 0px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
div#pager .ui-resizable-handle {
  top: 0px;
  height: 8px;
  background: #f7f7f7;
  border-top: 1px solid #cfcfcf;
  border-bottom: 1px solid #cfcfcf;
  /* This injects handle bars (a short, wide = symbol) for 
        the resize handle. */
}
div#pager .ui-resizable-handle::after {
  content: '';
  top: 2px;
  left: 50%;
  height: 3px;
  width: 30px;
  margin-left: -15px;
  position: absolute;
  border-top: 1px solid #cfcfcf;
}
.quickhelp {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
  line-height: 1.8em;
}
.shortcut_key {
  display: inline-block;
  width: 21ex;
  text-align: right;
  font-family: monospace;
}
.shortcut_descr {
  display: inline-block;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
span.save_widget {
  height: 30px;
  margin-top: 4px;
  display: flex;
  justify-content: flex-start;
  align-items: baseline;
  width: 50%;
  flex: 1;
}
span.save_widget span.filename {
  height: 100%;
  line-height: 1em;
  margin-left: 16px;
  border: none;
  font-size: 146.5%;
  text-overflow: ellipsis;
  overflow: hidden;
  white-space: nowrap;
  border-radius: 2px;
}
span.save_widget span.filename:hover {
  background-color: #e6e6e6;
}
[dir="rtl"] span.save_widget.pull-left {
  float: right !important;
  float: right;
}
[dir="rtl"] span.save_widget span.filename {
  margin-left: 0;
  margin-right: 16px;
}
span.checkpoint_status,
span.autosave_status {
  font-size: small;
  white-space: nowrap;
  padding: 0 5px;
}
@media (max-width: 767px) {
  span.save_widget {
    font-size: small;
    padding: 0 0 0 5px;
  }
  span.checkpoint_status,
  span.autosave_status {
    display: none;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  span.checkpoint_status {
    display: none;
  }
  span.autosave_status {
    font-size: x-small;
  }
}
.toolbar {
  padding: 0px;
  margin-left: -5px;
  margin-top: 2px;
  margin-bottom: 5px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
.toolbar select,
.toolbar label {
  width: auto;
  vertical-align: middle;
  margin-right: 2px;
  margin-bottom: 0px;
  display: inline;
  font-size: 92%;
  margin-left: 0.3em;
  margin-right: 0.3em;
  padding: 0px;
  padding-top: 3px;
}
.toolbar .btn {
  padding: 2px 8px;
}
.toolbar .btn-group {
  margin-top: 0px;
  margin-left: 5px;
}
.toolbar-btn-label {
  margin-left: 6px;
}
#maintoolbar {
  margin-bottom: -3px;
  margin-top: -8px;
  border: 0px;
  min-height: 27px;
  margin-left: 0px;
  padding-top: 11px;
  padding-bottom: 3px;
}
#maintoolbar .navbar-text {
  float: none;
  vertical-align: middle;
  text-align: right;
  margin-left: 5px;
  margin-right: 0px;
  margin-top: 0px;
}
.select-xs {
  height: 24px;
}
[dir="rtl"] .btn-group > .btn,
.btn-group-vertical > .btn {
  float: right;
}
.pulse,
.dropdown-menu > li > a.pulse,
li.pulse > a.dropdown-toggle,
li.pulse.open > a.dropdown-toggle {
  background-color: #F37626;
  color: white;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
/** WARNING IF YOU ARE EDITTING THIS FILE, if this is a .css file, It has a lot
 * of chance of beeing generated from the ../less/[samename].less file, you can
 * try to get back the less file by reverting somme commit in history
 **/
/*
 * We'll try to get something pretty, so we
 * have some strange css to have the scroll bar on
 * the left with fix button on the top right of the tooltip
 */
@-moz-keyframes fadeOut {
  from {
    opacity: 1;
  }
  to {
    opacity: 0;
  }
}
@-webkit-keyframes fadeOut {
  from {
    opacity: 1;
  }
  to {
    opacity: 0;
  }
}
@-moz-keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
@-webkit-keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
/*properties of tooltip after "expand"*/
.bigtooltip {
  overflow: auto;
  height: 200px;
  -webkit-transition-property: height;
  -webkit-transition-duration: 500ms;
  -moz-transition-property: height;
  -moz-transition-duration: 500ms;
  transition-property: height;
  transition-duration: 500ms;
}
/*properties of tooltip before "expand"*/
.smalltooltip {
  -webkit-transition-property: height;
  -webkit-transition-duration: 500ms;
  -moz-transition-property: height;
  -moz-transition-duration: 500ms;
  transition-property: height;
  transition-duration: 500ms;
  text-overflow: ellipsis;
  overflow: hidden;
  height: 80px;
}
.tooltipbuttons {
  position: absolute;
  padding-right: 15px;
  top: 0px;
  right: 0px;
}
.tooltiptext {
  /*avoid the button to overlap on some docstring*/
  padding-right: 30px;
}
.ipython_tooltip {
  max-width: 700px;
  /*fade-in animation when inserted*/
  -webkit-animation: fadeOut 400ms;
  -moz-animation: fadeOut 400ms;
  animation: fadeOut 400ms;
  -webkit-animation: fadeIn 400ms;
  -moz-animation: fadeIn 400ms;
  animation: fadeIn 400ms;
  vertical-align: middle;
  background-color: #f7f7f7;
  overflow: visible;
  border: #ababab 1px solid;
  outline: none;
  padding: 3px;
  margin: 0px;
  padding-left: 7px;
  font-family: monospace;
  min-height: 50px;
  -moz-box-shadow: 0px 6px 10px -1px #adadad;
  -webkit-box-shadow: 0px 6px 10px -1px #adadad;
  box-shadow: 0px 6px 10px -1px #adadad;
  border-radius: 2px;
  position: absolute;
  z-index: 1000;
}
.ipython_tooltip a {
  float: right;
}
.ipython_tooltip .tooltiptext pre {
  border: 0;
  border-radius: 0;
  font-size: 100%;
  background-color: #f7f7f7;
}
.pretooltiparrow {
  left: 0px;
  margin: 0px;
  top: -16px;
  width: 40px;
  height: 16px;
  overflow: hidden;
  position: absolute;
}
.pretooltiparrow:before {
  background-color: #f7f7f7;
  border: 1px #ababab solid;
  z-index: 11;
  content: "";
  position: absolute;
  left: 15px;
  top: 10px;
  width: 25px;
  height: 25px;
  -webkit-transform: rotate(45deg);
  -moz-transform: rotate(45deg);
  -ms-transform: rotate(45deg);
  -o-transform: rotate(45deg);
}
ul.typeahead-list i {
  margin-left: -10px;
  width: 18px;
}
[dir="rtl"] ul.typeahead-list i {
  margin-left: 0;
  margin-right: -10px;
}
ul.typeahead-list {
  max-height: 80vh;
  overflow: auto;
}
ul.typeahead-list > li > a {
  /** Firefox bug **/
  /* see https://github.com/jupyter/notebook/issues/559 */
  white-space: normal;
}
ul.typeahead-list  > li > a.pull-right {
  float: left !important;
  float: left;
}
[dir="rtl"] .typeahead-list {
  text-align: right;
}
.cmd-palette .modal-body {
  padding: 7px;
}
.cmd-palette form {
  background: white;
}
.cmd-palette input {
  outline: none;
}
.no-shortcut {
  min-width: 20px;
  color: transparent;
}
[dir="rtl"] .no-shortcut.pull-right {
  float: left !important;
  float: left;
}
[dir="rtl"] .command-shortcut.pull-right {
  float: left !important;
  float: left;
}
.command-shortcut:before {
  content: "(command mode)";
  padding-right: 3px;
  color: #777777;
}
.edit-shortcut:before {
  content: "(edit)";
  padding-right: 3px;
  color: #777777;
}
[dir="rtl"] .edit-shortcut.pull-right {
  float: left !important;
  float: left;
}
#find-and-replace #replace-preview .match,
#find-and-replace #replace-preview .insert {
  background-color: #BBDEFB;
  border-color: #90CAF9;
  border-style: solid;
  border-width: 1px;
  border-radius: 0px;
}
[dir="ltr"] #find-and-replace .input-group-btn + .form-control {
  border-left: none;
}
[dir="rtl"] #find-and-replace .input-group-btn + .form-control {
  border-right: none;
}
#find-and-replace #replace-preview .replace .match {
  background-color: #FFCDD2;
  border-color: #EF9A9A;
  border-radius: 0px;
}
#find-and-replace #replace-preview .replace .insert {
  background-color: #C8E6C9;
  border-color: #A5D6A7;
  border-radius: 0px;
}
#find-and-replace #replace-preview {
  max-height: 60vh;
  overflow: auto;
}
#find-and-replace #replace-preview pre {
  padding: 5px 10px;
}
.terminal-app {
  background: #EEE;
}
.terminal-app #header {
  background: #fff;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
.terminal-app .terminal {
  width: 100%;
  float: left;
  font-family: monospace;
  color: white;
  background: black;
  padding: 0.4em;
  border-radius: 2px;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.4);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.4);
}
.terminal-app .terminal,
.terminal-app .terminal dummy-screen {
  line-height: 1em;
  font-size: 14px;
}
.terminal-app .terminal .xterm-rows {
  padding: 10px;
}
.terminal-app .terminal-cursor {
  color: black;
  background: white;
}
.terminal-app #terminado-container {
  margin-top: 20px;
}
/*# sourceMappingURL=style.min.css.map */
    </style>
<style type="text/css">
    .highlight .hll { background-color: #ffffcc }
.highlight  { background: #f8f8f8; }
.highlight .c { color: #408080; font-style: italic } /* Comment */
.highlight .err { border: 1px solid #FF0000 } /* Error */
.highlight .k { color: #008000; font-weight: bold } /* Keyword */
.highlight .o { color: #666666 } /* Operator */
.highlight .ch { color: #408080; font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: #408080; font-style: italic } /* Comment.Multiline */
.highlight .cp { color: #BC7A00 } /* Comment.Preproc */
.highlight .cpf { color: #408080; font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: #408080; font-style: italic } /* Comment.Single */
.highlight .cs { color: #408080; font-style: italic } /* Comment.Special */
.highlight .gd { color: #A00000 } /* Generic.Deleted */
.highlight .ge { font-style: italic } /* Generic.Emph */
.highlight .gr { color: #FF0000 } /* Generic.Error */
.highlight .gh { color: #000080; font-weight: bold } /* Generic.Heading */
.highlight .gi { color: #00A000 } /* Generic.Inserted */
.highlight .go { color: #888888 } /* Generic.Output */
.highlight .gp { color: #000080; font-weight: bold } /* Generic.Prompt */
.highlight .gs { font-weight: bold } /* Generic.Strong */
.highlight .gu { color: #800080; font-weight: bold } /* Generic.Subheading */
.highlight .gt { color: #0044DD } /* Generic.Traceback */
.highlight .kc { color: #008000; font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: #008000 } /* Keyword.Pseudo */
.highlight .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: #B00040 } /* Keyword.Type */
.highlight .m { color: #666666 } /* Literal.Number */
.highlight .s { color: #BA2121 } /* Literal.String */
.highlight .na { color: #7D9029 } /* Name.Attribute */
.highlight .nb { color: #008000 } /* Name.Builtin */
.highlight .nc { color: #0000FF; font-weight: bold } /* Name.Class */
.highlight .no { color: #880000 } /* Name.Constant */
.highlight .nd { color: #AA22FF } /* Name.Decorator */
.highlight .ni { color: #999999; font-weight: bold } /* Name.Entity */
.highlight .ne { color: #D2413A; font-weight: bold } /* Name.Exception */
.highlight .nf { color: #0000FF } /* Name.Function */
.highlight .nl { color: #A0A000 } /* Name.Label */
.highlight .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */
.highlight .nt { color: #008000; font-weight: bold } /* Name.Tag */
.highlight .nv { color: #19177C } /* Name.Variable */
.highlight .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */
.highlight .w { color: #bbbbbb } /* Text.Whitespace */
.highlight .mb { color: #666666 } /* Literal.Number.Bin */
.highlight .mf { color: #666666 } /* Literal.Number.Float */
.highlight .mh { color: #666666 } /* Literal.Number.Hex */
.highlight .mi { color: #666666 } /* Literal.Number.Integer */
.highlight .mo { color: #666666 } /* Literal.Number.Oct */
.highlight .sa { color: #BA2121 } /* Literal.String.Affix */
.highlight .sb { color: #BA2121 } /* Literal.String.Backtick */
.highlight .sc { color: #BA2121 } /* Literal.String.Char */
.highlight .dl { color: #BA2121 } /* Literal.String.Delimiter */
.highlight .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */
.highlight .s2 { color: #BA2121 } /* Literal.String.Double */
.highlight .se { color: #BB6622; font-weight: bold } /* Literal.String.Escape */
.highlight .sh { color: #BA2121 } /* Literal.String.Heredoc */
.highlight .si { color: #BB6688; font-weight: bold } /* Literal.String.Interpol */
.highlight .sx { color: #008000 } /* Literal.String.Other */
.highlight .sr { color: #BB6688 } /* Literal.String.Regex */
.highlight .s1 { color: #BA2121 } /* Literal.String.Single */
.highlight .ss { color: #19177C } /* Literal.String.Symbol */
.highlight .bp { color: #008000 } /* Name.Builtin.Pseudo */
.highlight .fm { color: #0000FF } /* Name.Function.Magic */
.highlight .vc { color: #19177C } /* Name.Variable.Class */
.highlight .vg { color: #19177C } /* Name.Variable.Global */
.highlight .vi { color: #19177C } /* Name.Variable.Instance */
.highlight .vm { color: #19177C } /* Name.Variable.Magic */
.highlight .il { color: #666666 } /* Literal.Number.Integer.Long */
    </style>
<style type="text/css">
    
/* Temporary definitions which will become obsolete with Notebook release 5.0 */
.ansi-black-fg { color: #3E424D; }
.ansi-black-bg { background-color: #3E424D; }
.ansi-black-intense-fg { color: #282C36; }
.ansi-black-intense-bg { background-color: #282C36; }
.ansi-red-fg { color: #E75C58; }
.ansi-red-bg { background-color: #E75C58; }
.ansi-red-intense-fg { color: #B22B31; }
.ansi-red-intense-bg { background-color: #B22B31; }
.ansi-green-fg { color: #00A250; }
.ansi-green-bg { background-color: #00A250; }
.ansi-green-intense-fg { color: #007427; }
.ansi-green-intense-bg { background-color: #007427; }
.ansi-yellow-fg { color: #DDB62B; }
.ansi-yellow-bg { background-color: #DDB62B; }
.ansi-yellow-intense-fg { color: #B27D12; }
.ansi-yellow-intense-bg { background-color: #B27D12; }
.ansi-blue-fg { color: #208FFB; }
.ansi-blue-bg { background-color: #208FFB; }
.ansi-blue-intense-fg { color: #0065CA; }
.ansi-blue-intense-bg { background-color: #0065CA; }
.ansi-magenta-fg { color: #D160C4; }
.ansi-magenta-bg { background-color: #D160C4; }
.ansi-magenta-intense-fg { color: #A03196; }
.ansi-magenta-intense-bg { background-color: #A03196; }
.ansi-cyan-fg { color: #60C6C8; }
.ansi-cyan-bg { background-color: #60C6C8; }
.ansi-cyan-intense-fg { color: #258F8F; }
.ansi-cyan-intense-bg { background-color: #258F8F; }
.ansi-white-fg { color: #C5C1B4; }
.ansi-white-bg { background-color: #C5C1B4; }
.ansi-white-intense-fg { color: #A1A6B2; }
.ansi-white-intense-bg { background-color: #A1A6B2; }

.ansi-bold { font-weight: bold; }

    </style>


<style type="text/css">
/* Overrides of notebook CSS for static HTML export */
body {
  overflow: visible;
  padding: 8px;
}

div#notebook {
  overflow: visible;
  border-top: none;
}@media print {
  div.cell {
    display: block;
    page-break-inside: avoid;
  } 
  div.output_wrapper { 
    display: block;
    page-break-inside: avoid; 
  }
  div.output { 
    display: block;
    page-break-inside: avoid; 
  }
}
</style>

<!-- Custom stylesheet, it must be in the same directory as the html file -->
<link rel="stylesheet" href="custom.css">

<!-- Loading mathjax macro -->
<!-- Load mathjax -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-AMS_HTML"></script>
    <!-- MathJax configuration -->
    <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
            inlineMath: [ ['$','$'], ["\\(","\\)"] ],
            displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
            processEscapes: true,
            processEnvironments: true
        },
        // Center justify equations in code and markdown cells. Elsewhere
        // we use CSS to left justify single line equations in code cells.
        displayAlign: 'center',
        "HTML-CSS": {
            styles: {'.MathJax_Display': {"margin": 0}},
            linebreaks: { automatic: true }
        }
    });
    </script>
    <!-- End of mathjax configuration --></head>
<body>
  <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">

<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Analysing-titanic-data-and-predicting-Survivors-based-on-Passenger-class,-Sex,-Fare-and-Embarked-location">Analysing titanic data and predicting Survivors based on Passenger class, Sex, Fare and Embarked location<a class="anchor-link" href="#Analysing-titanic-data-and-predicting-Survivors-based-on-Passenger-class,-Sex,-Fare-and-Embarked-location">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="This-is-an-attempt-to-participate-Kaggle's-Machine-Learning-Prediction-compitition-(https://www.kaggle.com/c/titanic/data)">This is an attempt to participate Kaggle's Machine Learning Prediction compitition (<a href="https://www.kaggle.com/c/titanic/data">https://www.kaggle.com/c/titanic/data</a>)<a class="anchor-link" href="#This-is-an-attempt-to-participate-Kaggle's-Machine-Learning-Prediction-compitition-(https://www.kaggle.com/c/titanic/data)">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[50]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># import dependencies</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">import</span> <span class="nn">seaborn</span> <span class="k">as</span> <span class="nn">sns</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="kn">import</span> <span class="nn">math</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="k">import</span> <span class="n">train_test_split</span>
<span class="kn">from</span> <span class="nn">sklearn.linear_model</span> <span class="k">import</span> <span class="n">LogisticRegression</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Load dataset</span>
<span class="n">titanic_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">&#39;data/train.csv&#39;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Set default figure size</span>
<span class="n">sns</span><span class="o">.</span><span class="n">set</span><span class="p">(</span><span class="n">rc</span><span class="o">=</span><span class="p">{</span><span class="s1">&#39;figure.figsize&#39;</span><span class="p">:(</span><span class="mi">12</span><span class="p">,</span><span class="mi">8</span><span class="p">)})</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># QUick look at the data</span>
<span class="n">titanic_df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[4]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># total num of passengers </span>
<span class="nb">print</span><span class="p">(</span><span class="n">f</span><span class="s1">&#39;There are {len(titanic_df)} total passengers in this training dataset&#39;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>There are 891 total passengers in this training dataset
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Data-Cleanup">Data Cleanup<a class="anchor-link" href="#Data-Cleanup">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Looking at column types and count</span>
<span class="n">titanic_df</span><span class="o">.</span><span class="n">info</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;class &#39;pandas.core.frame.DataFrame&#39;&gt;
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#Check for null values in each column</span>
<span class="n">titanic_df</span><span class="o">.</span><span class="n">isnull</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[7]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Heatmap of null column values to get a better idea. Dark blue stands for null, </span>
<span class="c1"># and off white means no null value ofr the respective column for each point.</span>
<span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">titanic_df</span><span class="o">.</span><span class="n">isnull</span><span class="p">(),</span><span class="n">cmap</span><span class="o">=</span><span class="s2">&quot;YlGnBu&quot;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[8]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x118cf1f28&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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EXGoMkckV69eTWFhIXFxcZZzISEhxMXFMXHiRAoLC+nYsSPdunUDrm/54+XlRaVKlYyLWkRExAVodM+NuOaApEokioiIOIqmtu8Wx5dIfGjIu4a0++26QYa0ay2VSBQRERExmjsvthERERGR38FFF9sokRQRTa+JiIhNbK61vX79etatW4fZbLacM5lMnDhxghkzZnD58mVat27N7Nmz8fRUvipSnikpExExmGsOSJa9/c+NtbZTUlI4duwYCQkJJCQk8N5775GWlsahQ4f47LPPAJgyZQozZ85kx44dmM1mNm3aZHgnREREROTus6nWtslk4v3338fLy4sLFy6Ql5eHj48PP/74IwUFBbRo0QKAvn37smzZMgYPHmxsL0Tkd9HUtoiIwdx1sc3tam2/++67eHl5sWnTJhYsWEBAQACNGzfm2LFjpWpw+/r6lqrBLSLlk5IyERGDuWgiWebU9q9OnTrFqFGjiIyMpEGDBgAMGDCAzz//nFq1arFixQquXbtWqub2zTW4RURERMR1WJVIHjx4kBEjRvDSSy/Rp08fMjMzOXjwIACenp4EBQXx1VdfUadOnVI1uM+dO1eqBreIiIiIOzKbjHk5mk21tnNzc5kyZQo5OTmYzWZ27NhBq1ateOCBB6hUqZIlydy6daulBreIiIiIuBaba22/+OKLhISE4OHhQevWrRk5ciQAixYtIjo6mry8PJo2bcqwYcOMi15E7EKLbUREDOaiz0iq1raIiIiD6I+4u8XxtbYbvrjZkHYz4vsb0q61tFO4iIiIiNFcdPGxEkkRERERo7no1LbNJRKnTZvGwYMHqVKlCgATJkygc+fOv1k6UUREREpzzmlikf8oM5G8sUSiyWRi9OjRfPTRRxw9epS1a9eW2t7nhx9+ICEhgZSUFCpVqsSQIUP47LPPaN++vaGdEJHfR89piTiG7j03YvXO3c7FphKJZ8+e5ezZs0yfPp2srCw6d+7MhAkT8Pf3v23pRBEp3/TDIOIYuvfE2dlUInHdunXs37+fmJgY7r33XsaOHcvmzZsZMGDAbUsnioiIyK00IulGXPQxP6u3/zl16hRjx45l4sSJ9OnTp9R7H330ESkpKaxcudJyrri4mGnTplG3bl0mT55sZTja/kfEEfRjJuIYuvfulnKw/U94iiHtZizrbUi71rJqsc3BgwcJDw9n+vTplnKIp0+fpmvXrsD1mtqenp5kZmZy9uxZWrVqZSmd+O677xraARH5/Zzzh0FERBytzETy1xKJS5YsoV27dsD1xHH+/Pm0bdsWb29vNm7cSJ8+fSylE1NSUrj33nstpRNFpHzTqIiIiLHMLjq1/btKJA4aNIji4mK6dOlCcHAwwG+WThSR8ktJmYhj6N4TZ6cSiSKiEUkRB9G9d7c4/hnJByO2GtLu/y3pZUi71lJlGxFx0h8GEREn4s6VbURERMT+9EecODubSiS2adOGxYsXW97PysqiefPmvPXWW2RkZBATE8OlS5fw9fVl8eLFVKtWzZjoRcQuNL0m4hi699yIuy62uV2JxJYtW7J16/W5/uzsbAYNGsS0adMwm82EhYUxY8YMOnTowKJFi4iPj2fKlCmGd0REbKcfBhERsYXNJRJ/tXDhQkJCQmjQoAFHjx7F29ubDh06ABAaGkpOTo5BoYuIvWhURETEYC76jGSZJcQfeeQRWrRoAfynRGLHjh0tx/v372fYsGEAfP/999SqVYvp06fTp08fYmJi8Pb2NjB8EbEHJWUijqF7z42YDHo5mNWLbX4tkRgZGUmDBg0A2LhxI4MHD7aMVhYXF7N//37Wrl1Ls2bNWLp0KXFxcaX2oBSR8kk/aCJ3nzPOBojcqMwRSbheInHEiBG89NJLpeps79q1i8DAQMuxr68v9evXp1mzZgAEBwdz5MgRO4csIiIi4lzMFUyGvBytzETy1xKJixYtIigoyHL+/PnzFBQU4O/vbznXsmVLzp8/z8mTJwHYvXs3TZs2NSBsEREREXE0m0skNm3alDp16pS6tnLlyqxcuZLo6Gjy8/OpU6cOCxcutH/UIiIiIs6kHIweGkElEkVERBzEGZ+RdM7nqR1fIrHBjA8Maff0vMCyLzKQKtuIiIiIGM1dNyQXERERYzjn6J7YxKrlzc7HphKJkZGRJCUlsWrVKjw8PGjTpg1RUVF4enpy5MgRZs+ezdWrV7n//vt55ZVX8PX1NbQTIiIizkhT2+LsbCqRGB8fz9q1a9m8eTN+fn7MmjWLxMRERowYQXh4OHFxcbRt25YPPviAl19+mTfffPNu9EVEbKQfMxERg7nr1PbtSiRevXqVFi1a4OfnB8AzzzxDfHw8vXr1oqCggLZt21rOR0ZGcvXqVcvnRaT8UVIm4hi698TZ2VQiMTAwkMOHD5OZmUlJSQnp6emcO3eO++67D29vb/bs2QPA+++/T1FRERcuXDC2FyIiIiLlWQWTMS8Hs6lEYsOGDXnppZcICwujcuXKdOvWjS+//BKTycSyZctYsGABixYtolevXlSvXh0vLy8j+yAiv5OmtkUcQ/eeGykHSZ8RrEokDx48SHh4ONOnTycoKIjCwkICAgJISUkBYPv27ZYKN56eniQmJgLwyy+/8Prrr1O9enWDwhcREXFeSsrE2ZWZSP5aInHJkiW0a9cOgCtXrjBixAi2bdtGxYoVWbt2LSEhIQBMnz6dWbNmERAQwJo1a+jWrRsVKrjomncRF6EfMxERY5nddbHNb5VIHD9+PAMHDqS4uJjg4GB69OgBwKxZs4iJiSE/P59GjRoxb94846IXEREREYdRiUQRERFxcY4vkVh//keGtPvd9M6GtGstVbYRERERMZq7Tm2LiIiIMbRqW5ydEkkRERERo7nz9j+vvfYaO3bswGQy0b9/f0aOHMnGjRtJTEzEZDLx+OOPM3v2bCpWrMjOnTtZvnw5ZrOZevXqERsbS7Vq1Yzuh4iIiIjcZWXuy7N//37++c9/kpqaypYtW0hMTCQjI4PVq1ezYcMGUlNTuXbtGuvXrycvL49Zs2YRHx9PamoqjRo1Yvny5XejHyIiIiLll4tWtikzkXzyySf5+9//jqenJ7/88gslJSVUqlSJmJgYqlatislk4tFHH+Xs2bMUFRURExND7dq1AWjUqBGZmZmGd0JERERE7j6rpra9vLxYtmwZb7/9Nt26deP+++/ngQceAOD8+fOsW7eO2NhY7rvvPjp3vr4MvaCggPj4eIYOHWpc9CJiF3rgX0TEYI4fPDSE1YttwsPDGTNmDKGhoWzatImBAweSlZXF6NGj6devH23atLFcm5uby/jx42ncuDF9+vQxJHARsR8lZSIixjKXg2loI5Q5tf3tt99y4sQJAKpUqUKXLl346quv+PbbbwkJCaFPnz6MHz/ecv3PP//M4MGDVdVGRERExMWVOSJ55swZli1bxrvvvgvArl276NmzJ3/+85+ZNGkSvXv3tlxbUlJCaGgo3bt3Z9y4ccZFLSJ2paltERGDueuG5B07duTIkSP07t0bDw8PunTpwsWLFzl37hxr1qxhzZo1ADz77LM89thjHD9+nJKSEnbs2AHA448/rpFJkXJOSZmIY+jeE2enWtsiohFJEQfRvXe3OL7W9v+89okh7X7/l46GtGstVbYRESf9YRBxfrr33IhrzmwrkRQREXEUjUiKs7O5ROKv1q5dy44dO0hMTAQgOTmZv/71r9SsWROAp59+moiICANCFxERcW5KytxHhTL3yXFOZSaSN5ZILC4uJjAwkI4dO9KwYUO++eYb4uPjqV+/vuX6o0ePEhUVRXBwsKGBi4iIODuNSIqzs6lEore3N1evXmXmzJmEh4eXuv7LL78kOTmZHj168P/+3//j0qVLhgUvIiIi4gxMJmNejmbVQOuvJRKDgoJo1zNNNpAAACAASURBVK4dtWvX5q9//Sv9+vXD39+/1LW+vr6MGzeO1NRU6taty5w5cwwJXERERMRZuHUiCddLJO7bt4/MzEw2btxIZmYm/fr1u+W6lStX0qpVK0wmE6NHj+bTTz+1a8AiIiIiUj7YVCLx8OHDnDp1il69ehEdHc3Ro0eZNGkSubm5JCQkWD5rNpvx8PAwLHgRERERZ2AymQx53Ym0tDQCAwPp0qUL69atu+X9jIwMhg4daqlgaM3jiWUmkmfOnCE6OpqrV69y9epVdu3aRfv27dm+fTtbt27llVde4fHHH2fp0qV4e3uzatUqDh8+DFxf0d25c+c76qSIiIiI2FdWVhZLlixh/fr1pKSksHHjRr755hvL+2azmbCwMMaMGUNqaipNmjQhPj6+zHZtKpEYFBR022s9PDxYunQps2bNoqCggAYNGrBw4cI76KaIOIJWjoqIGMvRzzPu3buXtm3bUr16dQC6du1Keno6EyZMAODYsWN4e3vToUMHAEJDQ8nJySmzXZVIFBERERfn+BKJD7/5v4a0+6/BLW6b8Pn4+ODj42M5fuutt7hy5Yplb+/33nuPI0eOMHfuXAA++OADkpOT8fX15cSJEzRs2JCXX37Zknj+FlW2ERGNSIo4iO4992HUiOQ777zDihUrbjk/YcIEJk6caDm+du1aqWcqzWZzqePi4mL279/P2rVradasGUuXLiUuLo64uLj/+v1KJEVEPwwiDpL//WynTCblzpkMqmwzfPhw+vTpc8v5G0cjAerUqcOBAwcsx9nZ2fj5+VmOfX19qV+/Ps2aNQMgODj4lr3Cb8eqbr322msEBgYSFBTEmjVrADh06BADBgwgKCiIyZMnc/XqVeD6HHu/fv3o2bMnY8eOtWp+XURExB0piZTfy8fHh3r16t3yujmR/OMf/8i+ffs4f/48+fn5fPjhh5bnIQFatmzJ+fPnOXnyJAC7d++madOmZX5/mYnkjSUSt2zZQmJiIidPnmTixInMmTOH999/H4DNmzcDMG/ePMLDw0lNTeXBBx9k9erV1v/XEBEREXFBjt6QvHbt2kRERDBs2DB69+5NcHAwAQEBjBkzhi+//JLKlSuzcuVKoqOjCQoK4vPPPycqKqrMdsuc2r6xRGJWVhYlJSWcOHGCFi1a0LhxYwCio6MpKSkBrs/BX758GYD8/HyqVatmfS9FxCGccVRE0/EiInemR48e9OjRo9S5v/3tb5Z/N2/e3DIwaC2rV20vW7aMt99+m27dutGwYUO++eYbioqKyMjI4IknniAqKopKlSrx73//m1GjRuHt7U2VKlXYtGkT9913n5XhaNW2iIiI2JvjV203WW3Mqu0Tf+5Q9kUGuqPtf/Lz8wkNDeUPf/gD69evZ+PGjdx///3MmDGDBx54gDFjxtCvXz9iY2MJCAhgzZo17Nu3z6oNLa9TIikiIu5DswF3i+MTycfeNiaRPD7KsYmkTSUS4+Pjad68Of7+/nh4eNC9e3eOHDnC119/TaVKlQgICABg4MCB7N+/39geiIiIiIhDlPmM5JkzZ1i2bBnvvvsuALt27WLOnDksXryYzMxM6taty8cff0zTpk2pX78+P/30ExkZGTRs2JBdu3ZZlpGLSPmlUREREWM5urKNUWwqkdi7d2+qV69OaGgohYWFNGnShKlTp1KlShViY2OZNGkSZrOZmjVrMn/+/LvRDxH5HZSUiTiG7j1xdiqRKCIi4iCaDbhbHP+M5OMJnxrS7tERfzKkXWupso2IiIiIwYyqbONoLtotERERETGaVSOSr732Gjt27MBkMtG/f39GjhxJUlISq1atwsPDgzZt2hAVFYWn53+aW7p0KR4eHqUKhouIiIi4I1ddbGNTicSMjAyWLl1KQkICaWlpFBcXk5iYCEBubi7Tp0+31OQWEREREddUZiJ5Y4nEX375hZKSEo4cOUKLFi3w8/MD4JlnnmHnzp3A9e2BGjRowMiRI42NXERERMRJOLrWtlGsekbSy8uLZcuWERQURLt27QgICODw4cNkZmZSUlJCeno6586dA6B37968+OKLeHh4GBq4iIiIiDiW1au2w8PDGTNmDKGhoXzxxRe89NJLhIWFUblyZbp168aXX35pZJwiIiIuxzm30hFblIfRQyOUmUh+++23XL16lSZNmlhKJB45coQxY8aQkpICwPbt2/H39zc8WBEREVeifSTdRwUXTSTLnNo+c+YM0dHRXL16latXr7Jr1y7atGnDiBEjyMvL4+rVq6xdu5bAwMC7Ea+IiIiIlBM2lUjs2bMnhYWFDBw4kOLiYoKDg+nRo8fdiFdERETE6bjq1LZKJIqIiDiIprbvFseXSGz1rjElEg8OUolEEREREZfmqiOSSiRFREREDGZy0dU2qrUtIiIiIjaxekRywYIFXLhwgbi4OMu5yMhI2rZtS9++fUtde/z4cQYMGMDRo0ftF6mIGEbPaYmIGMutp7b37dtHcnIyTz/9NABZWVnExMSwb98+2rZtW+ra/Px85s6dS1FRkd2DFRFjKCkTcQzde+LsykwkL168yJIlSwgNDeXkyZMApKWl8dxzz1G9evVbro+Li2P48OH861//sn+0IiIiLkSzAe7DbUckZ86cSUREBJmZmZZzo0ePBuDgwYOlrt21axcFBQV069bNzmGKiIiIOC9XTST/62Kb9957j7p169KuXbsyG8rOzuaNN97g5ZdftltwIiIiIlJ+/dcRyQ8++IDs7Gx69erFpUuXuHLlCvPnz2f69Om3XPuPf/yDixcvMmTIEMu5Xr16sW7dOqpWrWr/yEXEbjS9JiJiLBfd/ee/J5Jr1qyx/DspKYn9+/ffNokEeP7553n++ectx40aNWLr1q12ClNEjKSkTEREbKENyUVEREQM5qrPSKrWtoiIiIPosZK7xfG1tp9K3mNIu5/1aW9Iu9bSiKSI6MdMRMRgJhetJahEUkSUlIk4iO499+GqU9t3lEjeWCbxo48+YtmyZVy7do1mzZoxZ84ccnNzGTVqlOX63NxcLly4wKFDh+weuIiIiLPTbIA4O6sTyRvLJF65coU5c+aQnJxMrVq1iIiIIDk5mYEDB1pWal+7do3hw4cTERFhWPAiIiLOTEmZ+zC56JCkVYnkzWUSvb292b17N15eXuTn5/PLL7/g4+NT6jNbtmyhSpUq9OjRw5DARUREnJ1GJMXZWZVI3q5MopeXF5988gmRkZH4+fnRvv1/Vg2VlJTw5ptv8vrrr9s/YhERERehpMx9uOiAZNmJ5I1lEpOSkkq917FjRz7//HMWL17MrFmz+Otf/wrAp59+SoMGDWjUqJExUYuIiLgAjUi6D7dNJG9XJnHatGkEBQVZRiF79OhR6lnInTt3EhgYaFzUIiIiIuJwZSaStyuTGBkZSWBgIFu2bOH+++8nPT2dJ554wnLdv//9b8aMGWNMxCIiIi5Co3vuw21HJG/nvvvuY+7cuYwdOxaTycTDDz/M7Nn/uRl++OEH6tSpY7cgRUREXJGmtsXZqUSiiIiIuDjHl0h8bvtnhrS7q/tThrRrLVW2ERGNiog4iO4991FBU9si4qr0wyDiGLr3xNnZXCIxIyODmJgYLl26hK+vL4sXL6ZatWqcPXuWKVOm8Msvv/Dggw+yaNEi7rnnHqPiFxERcVoakXQfFUzl6ElCO7L6Gcl9+/YRERHB008/TWxsLN26dWPGjBl06NCBRYsWYTabmTJlCmPHjqVnz54EBQWxcuVKrly5wpQpU6wMR89IijiCfsxExLU5/hnJrjv2GNLujq7ty77IQDaVSDx27Bje3t506NABgNDQUHJycigqKuKLL75g5cqVAPTt25cXXnjhDhJJERER96E/4tyHWz8jeXOJxO+//55atWoxffp0Tpw4QcOGDXn55Ze5cOECVatWxdPzerO+vr5kZWUZF72I2IV+GERExBYVyrrgxhKJvyouLmb//v0MGjSI5ORk/P39iYuLw2w2Y7ppx82bj0VERETcTQWDXo5mU4nEkydPUr9+fZo1awZAcHAw4eHh1KhRg9zcXEpKSvDw8CA7Oxs/Pz/DOyEiIiJSnrnqYpsyk9k1a9awbds2tm7dSnh4OM8++ywbNmzg/PnznDx5EoDdu3fTtGlTvLy8aN26NR988AEAKSkplucoRURERMS12LSPZOXKlVm5ciXR0dHk5+dTp04dFi5cCEBMTAxRUVG88cYb1K1bl8WLF9s1YBERERFn46qLbVQiUURExEG0avtucfz2P712fmpIu1s7/cmQdq2lyjYiIiIO4pxJmdiiPCyMMYISSRHRqIiIiMFcdWrb6kTyxvKIx44dY+bMmRQVFVG3bl1effVVfHx8+Oabb4iOjubKlStUq1aNuLg4HnjgASPjFxE7UFIm4hj6I06cnVUjrfv27SM5OdlyPG/ePMLDw0lNTeXBBx9k9erVAMyePZtx48aRmppKYGCgFtqIiIiIACaT2ZCXo5U5InlzeUSAa9eucfnyZQDy8/OpVq0acH2rIE9PT65du8bZs2fx8fExMHQRsReNioiIiC3KTCRvLo8IEBUVxahRo5g/fz5VqlRh06ZN1xvz9CQnJ4fAwEAKCgpITEw0LnIRsRslZSIixnLLZyRvLI+YlJQEQEFBATNmzCAhIYGAgADWrFnD1KlTiY+PB8DHx4c9e/bwv//7v4SFhbFr1y48PDyM74mI2EwjkiIixnLLVdu3K4/4448/UqlSJQICAgAYOHAgr732muX67t27YzKZ6NChAwUFBVy6dIkaNWoY3xMRsZmSMhERscV/TSTXrFlj+XdSUhL79+9n2rRpdO/enYyMDBo2bMiuXbssNbfffvttPD096dKlC//85z+57777lESKOAGNSIqIGMtVa23f8T6S1apVIzY2lkmTJmE2m6lZsybz588HIC4ujpdffpmVK1dy7733smzZMrsHLCL2p6RMxDF074mzU4lEERERB9FswN3i+BKJL3zyiSHtru3Y0ZB2raXKNiIiIg7inEmZyH8okRQRERExmFuu2v7V0KFDOX/+PJ6e1y+fM2cO33//PW+88QbFxcUMHz6cIUOGAHDo0CFiY2O5fPkyjRo1Ii4ujooVKxrXAxERESelqW334Zb7SAKYzWZOnz7Nxx9/bEkks7KyiIiIICkpiYoVKxISEkKbNm2oU6cOEydOZNWqVTRu3JjJkyezefNmBg8ebHhHRMR2+jETERFblJlIZmRkADBq1CguXrzIgAEDuOeee2jbti3Vq1cHoGvXrqSnp/PII4/QokULGjduDEB0dDQlJSUGhi8i9qCkTMQxdO+5D7fd/icnJ4d27drx8ssvU1RUxLBhw+jevTu+vr6Wa/z8/Dhy5AgVK1bE29ubiIgIMjIyeOKJJ4iKijK0AyIiIs5KswHi7Mp89rNly5YsXLiQe++9lxo1atC/f3+WLVuGyfSfyX6z2YzJZKKkpIQ9e/YwefJkkpKSyM/Pt5ROFBEREXFXFUzGvBytzETywIED7Nu3z3JsNpt54IEHyM7OtpzLzs7Gz8+PWrVq0bx5c/z9/fHw8KB79+4cOXLEmMhFREREnEQFg16OVubUdm5uLsuWLWPDhg0UFRWRnJzMq6++ypQpUzh//jxVqlThww8/ZO7cufj6+rJ8+XIyMzOpW7cuH3/8MU2bNr0b/RAREXE6miYWZ1dmIvnMM89w+PBhevfuzbVr1xg8eDCtWrUiIiKCYcOGUVRURP/+/QkICACubw0UGhpKYWEhTZo0YerUqYZ3QkRExBnpGUn34aqLbVQiUURERFyc40skhn72sSHtvvnUM4a0ay1VthEREXEQjUi6j/KwMMYISiRFREQcREmZ+1AiKSIuS6MiIo6he0+cnc21tj/99FO2b98OQMeOHYmMjAQgKSmJVatW4eHhQZs2bYiKirJ8TkRERMQdlYeteoxgU63tvXv3smfPHpKTkzGZTIwePZqPPvqIhx56iKVLl7J582b8/PyYNWsWiYmJjBw50vCOiIjtNMIgIiK2KDNBvrHWds+ePVm7di2+vr5ERUVRsWJFvLy8eOihhzh79ixfffUVLVq0wM/PD7i+ddDOnTuN7YGIiIiT0h9x7qOCyWzI606kpaURGBhIly5dWLdu3W9e949//INnn33WqjZtqrX94IMP8tRTTwFw+vRptm/fzrvvvovZbCYuLo7MzEz8/PxIT0/n3LlzVnZPRBxFz2mJOIYz3nvinLKysliyZAlJSUlUrFiRkJAQ2rRpw8MPP1zqunPnzrFgwQKr27Wp1vYnn3wCwKlTpxg1ahSRkZE0aNCABx98kJdeeomwsDCGDBlCo0aN8PLyusOuioiIiLgWR9fa3rt3L23btqV69ep4e3vTtWtX0tPTb7kuOjqaCRMmWN1umSOSBw4coKioiHbt2gHXn5n09PTk4MGDhIeHM336dIKCggAoLCwkICCAlJQUALZv346/v7/VwYiIY2h0T0TEWEYttsnJySEnJ+eW8z4+Pvj4+FiOf/75Z3x9fS3Hfn5+HDlypNRn/v73v/PYY4/RvHlzq7+/zH7l5uaycOFCCgsLycvLIzk5mWeffZbx48ezaNEiSxIJcOXKFUaMGEFeXh5Xr15l7dq1BAYGWh2MiIiIiFjvnXfe4bnnnrvl9c4775S67tq1a5hM/xnCNJvNpY6//vprPvzwQ8aNG3dH329Tre309HQKCwuJi4uzXBcSEsKgQYMYP348AwcOpLi4mODgYHr06HFHAYmIiLgLzQa4D6M2JB8+fDh9+vS55fyNo5EAderU4cCBA5bj7Oxsy+JogPT0dLKzs+nXrx9FRUX8/PPPDB48mPXr1//X71etbREREQdxxsU2zpn8Or7WduT+3Ya0u/BJ61ZXZ2VlMWjQIDZv3kyVKlUICQlh7ty5BAQE3HLtmTNnGDZsGLt3lx2zq+6PKSIiIlJumExmQ17Wql27NhEREQwbNozevXsTHBxMQEAAY8aM4csvv7S9XxqRFBERcQyNSN4tjh+RnHZglyHtxrZ+zpB2rWVzicRFixbdcq558+asWLHitqUTRaT80o+ZiIjYwqYSibc7B79dOrFz587G9UBEfjclZSIixnLVZwnLTCRvLJF48eJFBgwYYNlT8sZzL7zwQqnSiYCldKKIiIiIuB6bSiReunTpv5ZNhNKlE0VERETc2Z3WxXYWZSaSLVu2pGXLlpbj/v37c/bsWRYuXFjq3CeffGJJJE+dOsXYsWMtpRNFRERE3JlR+0g6mk0lEk+ePMm+fftuKZsI3LZ0ooiUb1psIyIitigzkczNzWXZsmVs2LCBoqIikpOT6dq1KwsXLix1bvbs2WRmZjJ+/HiWLFliSTJFpPxTUiYiYiy3HZG8XYnE4cOHU1RUVOpcy5YteeWVV36zdKKIiIiIuBZtSC4imtoWcRDde3eL4zckf+XQTkPajW7ZyZB2rWXVhuQi4tqc84dBRMR5uOqqbVfdH1NEREREDGZTicSgoCDef/99y/tnzpyhV69ezJw5k2nTpnHw4EGqVKkCwIQJE1TZRkRERNya2y62+a1yiC+++CJwfc/I8ePHM2HCBACOHj3K2rVr8fPzMyhkERER16DHSsTZ2VQi8YUXXrC8P2vWLCIiIqhRowb5+fmcPXuW6dOnk5WVRefOnZkwYQIVKmgGXURE5GZabOM+XHVEsswM79cSiStXriQhIYENGzbw2WefAbB3714KCgro3r07AOfOnaNt27bMnz+fTZs2ceDAATZv3mxsD0RERETEIWwqkfhrOcQNGzYwcuRIy3v+/v6sXLnScjx06FBSUlIYMGCAncMWEXvSqIiIiLE8XHRE0qYSiZ6enly9epUvvvii1ObjX331FadPn6Zr166lrhWR8k1JmYiIsdx2ajs3N5eFCxdSWFhIXl4eycnJdO7cma+++ooGDRrg7e1tudZsNjN//nwuXbpEUVERGzdu1IptERERERdlU4nEli1b8sEHH1CnTp1S1zZu3JgXX3yRQYMGUVxcTJcuXQgODjYseBGxD01ti4gYy1U3JFeJRBEREQfRH3F3i+NLJL527END2v1L0y6GtGstPcAoIiIiYjBXfUZSiaSIiIiIwTwcHYBBbCqROGfOHD7//HO2bNlCxYoVCQwMJCwsDIBjx44xc+ZMioqKqFu3Lq+++io+Pj7G9UBEREREHKLMVdu/lkjcunWr5XX58mXS0tLYsmULKSkpHD58mA8/vD73P2/ePMLDw0lNTeXBBx9k9erVhndCREREpDyrYDLm5Wg2lUgsKCigffv2VK1aFYA//elP7Ny5ky5dunDt2jUuX74MQH5+PtWqVTMwfBGxBz3wLyIitrCpRGKjRo3Ys2cPFy9epLCwkN27d3Pu3DkAoqKiiI6Opn379uzdu5eQkBDDOyEiIiJSnlUwmQ15OZpNJRI//fRT+vbty9ChQ6levTrt2rXj8OHDFBQUMGPGDBISEggICGDNmjVMnTqV+Ph4QzshIr+PRvdERIylEok3lEgE6NKli6XO9qpVq/D39+frr7+mUqVKBAQEADBw4EBee+01o2IXETvR1LaIiNiizEQyNzeXZcuWsWHDBoqKikhOTiY6Oppx48axZcsW8vPz2bx5M/PmzaN+/fr89NNPZGRk0LBhQ3bt2kWzZs3uRj9E5HdQUiYiYqzysDDGCDaVSHzyySfp0qULPXv2pKSkhBEjRtCqVSsAYmNjmTRpEmazmZo1azJ//nzDOyEiIiIid59KJIqIiDiIHiu5WxxfIvGdUzsMaXf4I10NaddaZa7aFhERERG5HZVIFBGNiog4SP73s53y/pM757bPSIqI61NSJuIYSiLdh0c52PPRCFZNbe/evZu+ffvSvXt3XnnlFQA2btxIcHAwPXr0YNq0aVy9erXUZyIjI0lKSrJ/xCIiIiJSLpSZSP7www/ExMTw+uuvk5qayvHjx3nnnXdYvXo1GzZsIDU1lWvXrrF+/XoAsrKyCA0NZccOYx4qFREREXE2FQx6OVqZU9sfffQRgYGB1KlTB4AlS5ZQXFzMww8/bKm1/eijj3L27FkA0tLSeO6556hevbqBYYuIiDg/PVYizq7MRPK7777Dy8uL0NBQMjMzefrpp5k0aRIPPPAAAOfPn2fdunXExsYCMHr0aAAOHjxoYNgiIiLOzxmfkVTyaxu3XWxTUlLCgQMHSExMxNvbm7CwMJKTk+nbty9ZWVmMHj2afv360aZNm7sRr4iIiMtQUuY+XDWRLHN6vVatWrRr144aNWpQuXJlOnXqxJEjR/j2228JCQmhT58+jB8//m7EKiIiIiLliFUlEqdOnUpOTg733HMPn376Ke3atePPf/4zkyZNonfv3ncjThExkKbXRBxD9577cNXtf8pMJJs3b87o0aMZPHgwRUVFPPXUU5SUlHDu3DnWrFnDmjVrAHj22Wf5y1/+YnjAImJ/+mEQERFbqNa2iGhURMRBdO/dLY6vtb31u+2GtNurfndD2rWWKtuIiJP+MIiIOA+3XWwjIiIiInI7Vo1I7t69mxUrVpCfn89TTz1FdHS05b21a9eyY8cOEhMTS33m+PHjDBgwgKNHj9o3YhGxO02viYgYy1VHJMtMJH8tkfjee+9Rs2ZNhg8fzieffELHjh355ptviI+Pp379+qU+k5+fz9y5cykqKjIscBGxHyVlIo6he0+cnU0lEitVqsTVq1eZOXMm4eHhbN26tdRn4uLiGD58OP/617+MiVpE7EojkiKOoXvPfXi464jkb5VIjIuLo1+/ftSrV6/U9bt27aKgoIBu3boZFrSI2Jd+GEQcQ/eeODubSiS+9957ZGZmMm3aND7//HPLtdnZ2bzxxhskJCQYGbOIiIiIU6ngrhuS31giEaBTp04cOnSIU6dO0atXL65cucK5c+eYNGkSTz31FBcvXmTIkCGWz/fq1Yt169ZRtWpV43ohIiLihDS17T5cdZscm0okPvfcc8TGxgLw+eefs2LFCpYuXQrA888/b/lso0aNbnl+UkRERERcg00lEvv163c3YhORu0SjIiIixnLb7X8A+vfvT//+/W/7Xps2bWjTps1t3/vqq69sj0xEREREyjWVSBQRje6JiBjMbbf/EREREZHfx21XbcPtSyQeOnSI2NhYLl++TKNGjYiLi+Pbb78lKirK8rnz589TrVo1tm3bZlgHREREnJVmA8TZ2VQicefOncyaNYtVq1bRuHFjJk+ezObNmxk8eLBllXZ+fj7PP/88s2bNMroPIiIiTkkL3dyH2y62uV2JxH//+9+0aNGCxo0bAxAdHU1JSUmpz7311lv84Q9/oHXr1gaELSIiIiKOZlOJxHvuuQdvb28iIiLIyMjgiSeeKDWlnZuby6ZNm0hLSzM0eBERERFn4KojkmVutF5SUsK+ffuYP38+Gzdu5MiRI5SUlLBnzx4mT55MUlIS+fn5xMfHWz6TmppKp06dqFmzpqHBi4iIiDiDCga9HK3MGG4skVi5cmU6derEG2+8QfPmzfH398fDw4Pu3btz5MgRy2d27txJYGCgoYGLiIiIiGPZVCLxxRdfZNOmTWRmZlK3bl0+/vhjmjZtCoDZbObYsWO0bNnS8OBFxD70wL+IiLFMLjq1bVOJxHHjxvH4448TGhpKYWEhTZo0YerUqcD1LX+8vLyoVKmS4cGLiH0oKRMREVuYzGZzOdoh82tHByAiInLXaDbgbnnU0QHwRfb7hrT7B98gQ9q1lirbiIh+zERExCZKJEVESZmIiMFc9RlJq1aO7969m759+9K9e3deeeUVAJKSkggMDKRHjx688sorFBcXA3D27FmGDBlCt27dCAsL4/Lly8ZFLyIiIuIE3Hb7n19LJL7++uukpqZy/Phx3nnnHZYuXUpCQgJpaWkUFxeTmJgIwOzZsxk8eDDp6ek8/vjjvP7664Z3Zb0IkAAAIABJREFUQkRERETuvjITyRtLJHp5ebFkyRJ8fX1p0aIFfn5+wPUtgnbu3ElRURFffPEFXbt2BaBv376kp6cb2wMRERGRcs5kMhvycjSbSiT26tWLBQsWkJmZiZ+fH+np6Zw7d44LFy5QtWpVPD2vN+vr60tWVpbhnRAREXFGej5ZnF2ZiWRJSQkHDhwgMTERb29vwsLCqF+/Pi+99BJhYWFUrlyZbt268eWXX2I2mzHd9DTpzcciUv5o1baIY+jecx+umg2VmUjeWCIRoFOnTnzxxReMHTuWlJQUALZv346/vz81atQgNzeXkpISPDw8yM7Otkx/i0j5pR8GERFjueq4mk0lEp988klGjBjBtm3bqFixImvXriUkJAQvLy9at27NBx98QI8ePUhJSaFDhw53ox8i8jtoVETEMfK/n+2U95/Ir6yqbLN582YSEhIsJRKjo6PZsmULCQkJFBcXExwczMSJEwH48ccfiYqK4pdffqFu3bosXryYatWqWRmOKtuIiIiIvTm+ss2R89sMaTegRrAh7VpLJRJFRETExSmRNMr/b+/u42q+//+BP05XCmtWq2wxG6mGXF+EhjVXTdeUsOa6GmHNRXQhIim5rhnmc5sPUXSBzLRVxjaWMbYoZuvj11yFD12pqM77+4df51OS0hzv9zk97m7dbp33++z0aLfTOc/zer9eryc72xAREYlEFS9rc1pJ02g01zmSRERERPTPqGkd2XAhuX//fuzevVtx+9q1a3BycsLSpUtRUVGBGTNmYNasWRgwYACAxxuYb9q0CXK5HFZWVggNDYWOjo7yfgMiIiIVxdE9eplSUlKwZcsWVFZWYvLkyZg0aVKt82lpadi8eTMEQUC7du0QHh7e4DqX55ojeeXKFcyePRtxcXEoKChAQEAAsrOzsX37dgwYMAClpaUYNWoUkpOT8frrr8PPzw/W1tYYP358I38C50gSiYGX14jEwb+9l0X8OZLZBcqZI9mlTePmSObn52PChAlISkqCjo4OPDw8sG7dOpiZmQEASkpKMHr0aCQmJsLExAQbN25EcXExgoKCnvm4z3Vpe9myZfDz84OBgQG+/PJLzJgxAzt37lScb9myJTIyMqCtrY2ysjL897//hb6+/vP8CCIiomZDNYsyUkUnT56EtbU12rRpAwAYNWoUjh49Cl9fXwBARUUFQkJCYGJiAgCwsLBASkpKg4/b6ELy5MmTKC8vh52dHQBg0aJFAFCrkAQAbW1tHD9+HIsWLYKxsTFsbGwa+yOISCR8MyMSB0ckmw9lzZEsKipCUVFRneP6+vq1BvNu374NIyMjxW1jY2P8/vvvituvvfYaRowYAQAoLy/Htm3b4Onp2eDPb3QhGRcXh6lTpzbqvkOHDkVmZibWrVuHZcuWYe3atY39MURERM0GizL6p3bu3Ino6Og6x319fRV7fAOAXC6v1bb6aW2tAaC4uBizZ8+GpaUlXFxcGvz5jSokHz16hF9++QWrV69+5v0KCgpw4cIFxSikg4MD/Pz8GvMjiIiIiNSWskYkJ0+e/NSC78mphW3btsWZM2cUt5/Wxvr27duYPn06rK2tERAQ0Kif36hC8vLly3j77bfRsmXLZ95PEAQsXLgQiYmJePPNN3H06FH07t27UUGIiIiaG17abj6UtY/kk5ew6zNo0CBs3rwZ9+7dg56eHr799lusWLFCcb6qqgo+Pj6ws7PDrFmzGv3zG1VI/v3332jbtm2D93vttdewYsUKeHt7QyaTwczMDMuX8wlHREREJCYTExP4+fnh448/RkVFBcaNG4fu3btj5syZmDt3Lm7duoXs7GxUVVUhNTUVANCtWzeEhYU983HZIpGIiEgkHJF8WcTf/udKoXK2/+n8qrgtEjVE/elEREREpLLYIpGIOCpCRKRkMpmELgC/QCwkiYhFGRGRkrHX9v9X3WvbzMwMsbGxEAQBQ4cOxaJFiyCTyZCbm4uQkBAUFhbCyMgI69ata7BPIxERERGpngbnSLq5ueHgwYM4ePAgoqKiYGhoCGdnZ3z11VfYv38/UlJScO7cOfz0008QBAGffPIJZs6ciUOHDuHdd9/Ftm3bXsbvQURERCRZMplyvsTWpF7b3bt3x9dffw1tbW3cv38fJSUl0NfXx8WLF9GyZUsMGTIEAODj4/PUtj1EREREpPqa3GtbW1sb+/btQ0REBLp37w5LS0ukpaXh9ddfR0BAAHJyctCxY0cEBwcrLTwREZEq4/zk5kNdt8lp9D6Sc+fOxciRI2FvX3u/osrKSixZsgRvvPEGzMzMEBwcjN27d8PKygobNmzArVu3Gmyt+D/cR5JIDFy1TUTqTfx9JP9fSYpSHrdDawelPG5jNanX9s2bN3Hjxg306dMHWlpaGDNmDPbu3YuBAweiQ4cOsLKyAgDY29tj7ty5yktPRC8EizIiImqKJvXaLi4uxsKFC3HgwAG88sorSE1NRZ8+fdCrVy/cu3cPly5dgqWlJTIyMtC1a1el/gJERESqilcDmg8JrItRiib12jY3N4eXlxc8PDygqamJvn37YurUqdDW1kZMTAyCgoJQVlaGtm3bIjIyUmnhiYiIiEg87LVNREREak78OZJ/P1DOHMn2rVRgjiQRERG9eLy0TaqOhSQRERGRkjXbOZL1tUgsKyvD2bNnoaenBwDw9fXFiBEjkJSUhC+//BKampoYMGAAFi9eDC0t1qtERETUfGmoaSX5XHMkr1y5gtmzZyMuLg6TJ0/Gjh07YGxsrDifm5uLKVOmICEhAcbGxli2bBk6dOiAqVOnNvIncI4kERE1H7y0/bKIP0fyRqly5ki+2VLcOZLPtdF6dYtEPT093LhxAwEBAXBwcMCmTZsgl8tx+fJl9OzZU1Fcvv/++0hLS1NKcCIiIiJVIVPSl9ia1CLx77//hrW1NUJCQvDKK6/A29sbCQkJ6NevH1avXo2bN2/C2NgYR48exd27d5WZn4heAI6KEBFRUzS6kIyLi1Ncom7fvj1iYmIU5zw9PXHgwAG4u7tj/vz5+OSTT6Crq4vRo0cjKyvrxacmoheKRRkRkXLJZBLabfEFatSl7eoWiba2tgAed7pJTU1VnBcEAVpaWnj48CG6d++OAwcOIC4uDiYmJmjfvr1ykhMRERGpiGZ9afvJFomCIGDVqlWwtrZGy5YtER8fDxcXF5SWlmLKlCk4fPgwdHR0sHv3bnh4eCj1FyCif46XtomIqCma1CLR0tISXl5emDBhAiorKzFy5EjY29sDAGbPno3x48ejsrIS9vb2cHAQdzURERERkdhkUhg+VAK2SCQiIhIJrwa8LOJv/3O7/JBSHtdY11Epj9tY3CmciIhIJKpZlFFTqOmAJAtJIiIisXBEsvl4ro27VUiTWyTa2toiMjIScrkcXbp0wcqVK6Gjo4OzZ88iPDwcFRUVaNOmDVatWgVTU1Ol/hJE9M/wzYxIHHwek6prcotEFxcX/Otf/0KnTp0wd+5cvPfee3Bzc4OtrS0+//xzWFpaIiEhAenp6diyZUsjfwLnSBKJgYUkkTj4t/eyiD9H8t5D5cyRNGihQnMkq1skGhgYoKqqCiUlJaiqqsLDhw/RokULPHr0CPPmzYOlpSUAwMLCotZoJhFJk2q+MRCpPv7tkaprUotE4HFR6enpidatW6Ndu3YYPXo0dHR04OTkBACQy+WIjo7G8OHDlZOciF4YjooQiYN/e82Jei63aVKLxDt37iAqKgqHDx9Gu3btEB4ejvDwcISEPP6DePToERYvXozKykp4e3srJzkRvTB8YyAioqZoUovEM2fOwNzcHG+99RY0NDTg7u6O06dPAwAePHiAGTNmoLKyElu2bIG2trby0hMRERGpAJmS/omtSS0Szc3NERERgbt37+L1119Heno6rKysAAALFy5Ehw4dsHz5cmhoqOtidyIion+OVwOaD5lMPWuiJrVI7NSpE+bNm4ePP/4Ympqa6NChA0JDQ5GdnY309HSYmZnBxcUFAGBsbIzt27crJz0REZEK4xxJUnVskUhEfDMjEgn/9l4W8bf/KXj0jVIet42OnVIet7HY2YaIVPSNgYiIxMZCkoiIiEjJpLAwRhkaNfPz4MGDGDNmDMaMGYOIiAgAQE5ODlxdXTFq1CgEBgaisrISAJCcnAwbGxs4OTnByckJ69evV156IiIiIpUgU9KXuBockSwrK0NYWBiOHj0KfX19TJgwASdPnsSqVauwcuVK9OzZEwEBAdi3bx8mTpyICxcuYPHixbC3t38Z+YmIiIhIJA2OSFZVVUEul6OsrAyVlZWorKyElpYWysvL0bNnTwCAq6srjh49CgDIyspCcnIyHBwcsGDBAhQWFir3NyAiIiKSOJlMQylfYmswQevWrTFv3jzY2dlh6NChMDU1hba2NoyMjBT3MTIyQn5+vuL7WbNm4dChQ3jjjTcQGhqqvPREREREJJoGC8lLly4hMTERx44dww8//AANDQ389NNPkMn+d11eEATF7ZiYGPTp0wcymQwzZszADz/8oLz0RERERCqhmc6R/PHHHzFw4EAYGhoCeHwZe8eOHbhz547iPnfv3oWxsTGKi4uRmJiIKVOmAHhcYGpqaionORERkYrj1lvNh7qu2m6wkLS0tMSaNWtQWloKPT09ZGRkoH///khNTcXZs2fRp08fHDx4EEOGDEHLli3x5ZdfolevXujRowd2796NESNGvIzfg4iISOVwQ3JSdQ0WkjY2NsjOzoarqyu0tbVhZWUFLy8vjBgxAkFBQSgpKUHXrl0V7RI3bNiAZcuWoby8HG+//TYiIyNfxu9BREREJFnqOiLJFolExFERIlJz4rdILKnIUMrjtta2VcrjNhY72xARizIikfBDXHMi/lY9ysBCkoiIiEjJau52o07UszwmIiIiIqVrcq/ttLQ0ODk5wdHREbNmzVJ0sLl27RomTZoEJycneHp64vr168pLT0RERKQS1HMfyQYX25SVlWHo0KG1em37+PggJCQEiYmJMDExwcaNG1FcXIygoCAsXLgQvXr1wsSJE7Fr1y789ttviIqKamQcLrYhEgPnaRGJg397L4v4i20eVJ5QyuO20hqilMdtrCb12m7Tpg1CQkJgYmICALCwsMDNmzcBAHK5HCUlJQAeF6G6urpKjE9EREQkfTIl/RNbg4ttavba1tPTQ79+/dC7d2/FpNHy8nJs27YNnp6eAIB58+bBw8MDu3btQkVFBeLj45X7GxDRP6aaIwxERCS2BgvJmr22X3nlFSxYsAA7duzAjBkzUFxcjNmzZ8PS0hIuLi4AAH9/f4SGhmL48OFITU2Fr68vDh06pLarlYjUAS+vEREpm3qub25Sr+09e/bA0dER06dPh7W1NQICAgAA9+7dQ25uLoYPHw4AGDVqFEJCQnD//n0YGBgo8dcgon+CRRkRkXJJ4TK0MjRYHltaWuLkyZMoLS2FIAjIyMhAly5d4OPjAzs7OwQGBipGG1977TW0aNECZ86cAQCcPXsWrVq1YhFJREREpIaa1GvbzMwMX3zxBaqqqpCamgoA6NatG8LCwhAdHY0VK1agvLwcrVq1wubNm5X+SxARERFJmbpO8WOvbSIiIpFwfvLLIv72P+VVp5TyuLqaA5XyuI3FFolERERESqeeI5IsJImIiIiUTNZcV20Dj1skbtu2DQAwZMgQ+Pv7Izo6GomJidDX1wcAuLu7Y9KkSYr/ZtGiRbC2toarq6sSYhPRi8TLa0RE1BQNFpJlZWUICwur1SLx5MmTuHDhAtatW4devXrVun9+fj5CQkJw6tQpWFtbKy04Eb04LMqIiJStmV7artkisWXLlqisrESLFi1w4cIFbN26FdevX0e/fv3g7++PFi1aICUlBR988AHatGnzMvITERERkUgavGBfs0Xi0KFDYWpqCgsLC7z77rtYuHAhkpOTUVRUhM8//xwAMGPGDLi5uSk9OBEREZGqkMlkSvkSW4OFZM0WiT/88AM0NDQQFxeH7du3o1OnTtDS0sK0adNw/Pjxl5GXiIiISAXJlPQlrgYLyZotEnV0dODq6orjx48jISFBcR9BEKClxQXgRERERM1Jk1okmpmZYc2aNfj7778hCAJiY2MxYsSIl5GXiIiISOXIoKGUL7E1qUViYGAgrK2t8cknn6CiogK9e/fG1KlTX0ZeIiIiIpIItkgkIiIiNSd+i8QK+XmlPK62Rk+lPG5jcWIjERGRSNgMgFQdC0kiIiKRsChrPmQSWGGtDE1qkejo6IjFixcrzt+7dw+vvvoqDh8+rDiWnZ0Nd3d3XLhw4QVHJiIiUg8ckWw+pLDnozI0qUXie++9h4MHDyrOu7m5YdmyZbX+mxUrVqCiokJpwYmIiFQdizJSdU1ukVht69at6NevH/r27as4tnr1akyePBm//vqrclITERGpAY5INifib9WjDA0WkjVbJOrp6aFfv37o3bs3AKC4uBj79u1DSkqK4v7p6ekoLy/H6NGjlZeaiIiIiETXpBaJO3bsAAAcOnQIw4cPh6GhIQDgzp072LJlC4KDg5WbmoiIiEiFyJT0T2wNjkjWbJEIAK6urtizZw9mzJiBtLQ0eHt7K+77/fffo6CgAJMmTVIcc3JyQmxsLFq3bq2E+ERERKqLl4mbE/GLPmVosJC0tLTEmjVrUFpaCj09PWRkZMDKygqCIODixYvo1auX4r5ubm5wc3NT3LawsFAsyiEiIqLaOEeSVF2TWiR6eXnh3r170NbWrrXwhoiIiIjqUtftf9gikYiISCQckXxZxG+RKCBHKY8rw7uNvm9KSgq2bNmCyspKTJ48udZURADIyclBYGAgHjx4gL59+2L58uXQ0nr2mCMLSSIiIlJzUigkLyvlcWWwaNT98vPzMWHCBCQlJUFHRwceHh5Yt24dzMzMFPext7fHypUr0bNnTwQEBKBbt26YOHHiMx9XPTc1IiIiIpIQsVdtnzx5EtbW1mjTpg1atmyJUaNG4ejRo4rz169fR3l5OXr27Ang8eLqmufrw17bREREIuGlbfqnioqKUFRUVOe4vr4+9PX1Fbdv374NIyMjxW1jY2P8/vvv9Z43MjJCfn5+gz9fYoWk+EPPREREL0tZ3l6xI9BLo5waZ+fOzYiOjq5z3NfXF3PmzFHclsvltRb8CIJQ63ZD5+sjsUKSiIiIiBpr8uTJcHFxqXO85mgkALRt2xZnzpxR3L5z5w6MjY1rnb9z547i9t27d2udrw/nSBIRERGpKH19fbRr167O15OF5KBBg3Dq1Cncu3cPZWVl+PbbbzFkyBDFeVNTU7Ro0QJnz54FABw8eLDW+fpIbNU2ERERESlDSkoKtm7dioqKCowbNw4zZ87EzJkzMXfuXFhZWeHSpUsICgpCSUkJunbtivDwcOjo6DzzMVlIEhEREVGT8NI2ERERETUJC0kiIiIiahIWkkRERETUJCwkiYiIiKhJWEgSERERUZOwkCQipSgsLBQ7AhERKRkLSSJ6oXJycjB69Gg4OTkhPz8fI0aMwMWLF8WO1aDKykpcvHgRly5dgqrsilZaWoqbN2/ixo0bii+punLlSp1j58+fFyHJ8+OHIqL6qeU+kra2ts/sD5menv4S0zTsl19+eeb5fv36vaQkzy8vLw/nz5+Hg4MDli5diuzsbCxbtgxWVlZiR3uq69evIygoCNevX8fu3buxYMECrFq1Cu3atRM7Wr327t2LCRMmKG6XlZVhzZo1WLp0qYip6jdp0iSEhoZi/vz5OHDgAH766SesX78eCQkJYker108//QR/f38YGxtDLpejqKgIGzZsQPfu3cWOVq/o6Gjs2LEDr732muKYTCaT3Ovb2bNnIZfLERQUhLCwMEWRXllZiWXLliE1NVXkhPXLycmBn58fysvLER8fj48++ggbNmxA165dxY72TFeuXEFhYWGtD0RSfB9ZsmTJM8+Hh4e/pCT0T6hlr+1du3ZBEATExMSgffv2cHV1haamJlJSUnDt2jWx49WxadMmAEBBQQHy8vLQu3dvaGho4Ny5czA3N0dcXJzICeu3ZMkSuLm5IT09HVevXsWSJUsQFhYm2cxLly7F9OnTsXbtWhgZGcHe3h7+/v6IjY0VO1q90tLScOzYMYSHh+Ovv/5CcHAw3nvvPbFj1ausrAydOnVS3B48eDAiIiJETNSw8PBwfPnll7C0tAQAZGVlISQkBElJSSInq19SUhIyMjJqFZJSdPLkSZw+fRq3b9/Gxo0bFce1tLQwfvx4EZM1bOXKlYiJicH8+fNhYmKCZcuWISQkRNIfipYvX45jx46hffv2imMymQz//ve/RUz1dP379wcAHDt2DA8ePICjoyO0tLRw5MgRvPLKKyKno8ZSy0LS1NQUAHD58uVan2imTZsGV1dXsWLVa9euXQCAmTNnIjo6Gh06dADwePRMqqNO1R4+fAhnZ2cEBgbCwcEBffv2xaNHj8SOVa/79+/DxsYGUVFRkMlkcHd3l3QRCQA7duxAbGwsRo8eDV1dXXz++eeSHfEFgDZt2uDSpUuKqwKHDh3Cq6++KnKqZ9PR0VEUkQAk/f+3mrGxsUq82c6ZMwcAcODAATg7O4uc5vmo4oein376CUePHoWurq7YURrk4uICANizZw/i4+OhofF4tp2dnR3c3d3FjEbPQS0LyZpOnTqFgQMHAgCOHz8OTU1NkRPV78aNG4oiEgDefPNNSc95AgBNTU2kpqbi+++/x7x585CWlqZ4MZAiXV1d3Lp1S1HknDlzpsE+omL7+eefsWvXLowZMwb/+c9/sGXLFoSEhMDExETsaE+1bNky+Pv748qVK+jbty86dOiANWvWiB3rmfr27YvAwEC4u7tDU1MTX3/9NUxNTRXTTqR0WTA6OhoAoK+vj/Hjx2PIkCG1Xtd8fX3FivZMw4YNQ1BQEPLy8rBp0yZERERg8eLFkv6QoYofitq3b68yc3yrFRcXo6CgAAYGBgCAu3fvorS0VORU1FhqOUeyWnZ2Nvz9/XHnzh0IggBTU1NERkbCzMxM7GhPtWjRIshkMtjZ2UEQBKSkpKBVq1ZYsWKF2NHqdfnyZXz11VcYNmwYRo0aBT8/P3h7e9ca3ZGSrKwsxZvZW2+9hcLCQmzYsAE9e/YUO1q9bG1tsWrVKlhbWwMAYmNj8cUXX+CHH34QOdmzlZaWQi6Xo3Xr1mJHaZCnp2e956R2WbC6kKyPVAvJuXPnYvDgwYiNjUVCQgJiYmKQk5ODbdu2iR2tXnl5efD390dWVhZ0dXXRoUMHREVF4Z133hE7Wr0+++wznD9/Hr169ar1IVnK8w0PHDiAqKgo9O7dG4Ig4Pz58wgODsbIkSPFjkaNoNaFZLX79+9DJpOhTZs2Ykd5pkePHmH37t04ffo0AGDQoEGYOHEitLSkPXB8+/ZtGBsb48yZM7h8+TLGjh0r6csqFRUVuHr1KqqqqtCxY0fJj0g+ePAArVq1qnXs2rVrkl0g5OnpWWuxm0wmg66uLjp27AgfHx/Jj+ioisrKShw/fhwffPAB7t27h4yMDIwdO/aZCw3F5OrqiqSkJDg7O+PAgQMAAEdHRxw6dEjkZPWLi4uDh4eHSn0oSk5Ofurx6svIUnX79m2cO3cOMpkMffr0gaGhodiRqJGkXaE0kap+YtfR0cHIkSPRsWNH2NjY4ObNm5IvIkNCQlBRUYFp06Zh/vz5GDx4MM6dO4eoqCixoz1Vbm4u9u3bV2c7Dyl/Wi8oKICvr2+dleZSZWZmBi0tLYwdOxYAcPjwYdy6dQsmJiYIDAxs8O/zZZLL5dizZw/69+8Pc3Nz/Pvf/8b+/fvRpUsXBAcHS7pwCA4OhlwuxwcffAAAyMzMxO+//47Q0FCRkz2dpqYmiouLFYXu1atXJT0NBgB2794NDw8PtGzZUuwoDbpz5w6MjIwwYMAAsaM8t0ePHiEpKQm5ubkIDg7Gzp074eXlJfkP+fSYtKuUZubIkSPYsmULysvLFZ+EFy1aBCcnJ7Gj1SsrKwuJiYmIjo7GuHHjMGfOHEUBIUW+vr748MMPYWFhIXaURlO1lea//fZbrdXOlpaWGDt2LKKiohQjUVKxdu1a5ObmYtiwYTh79iw2btyIzZs34+LFi1ixYoWkF1ZcuHABKSkpAAADAwOsWbMGDg4OIqeq35w5c+Dp6YmbN29i1qxZOH/+vKQ/EAFA27Zt8fHHH6NHjx5o0aKF4rgUByOCgoKwdetWfPTRR5DJZLXmSUpxW6iaQkNDYWBggOzsbGhpaSEvLw8BAQGSHZCg2tSykKz+I68uxlTF9u3bsXfvXnz00UcwNDREcnIypk6dKulCsqqqCnK5HOnp6Vi+fDnKyspQVlYmdqx66evrS/JN4FlUbaV5RUUFrly5gs6dOwMA/vjjD8jlcpSXl6OiokLkdLWdOHECycnJ0NLSws6dOzFq1CgMGjQIgwYNgp2dndjxnkkulyumlQDAf//7X0mP8A0ZMgTdunXD77//jqqqKoSGhtaZsiE1Up47/aStW7cCADIyMkRO8vwuXryI5ORknDhxAnp6eoiIiJD0hyKqTS0LyWrVlyVUhYaGRq1LacbGxpJ+YwAAZ2dn2NjYoHfv3ujRowc+/PBDSW/b4OLigvXr18Pa2rrWtAEprcp9kqqtNA8KCsLMmTNhaGgIQRBQWFiINWvWYPPmzZL7UKShoaF4Hpw+fRre3t6Kc3K5XKxYjeLj4wMXFxf06dMHwOOR4MDAQJFT1S8qKgoLFizAsGHDAADff/89QkNDJV34PPmhUxAESe5FXFNxcTFiYmJw+vRpaGlpYdCgQfD29oaenp7Y0eolk8nw6NEjxWtc9boGUg1qXUiq0mUJAOjcuTN2796NyspK5OTkYM+ePZJd/Vxt6tSpmDx5sqLg3b17t2ILByk6d+4cfv31V/z666+KY1KoovZhAAATXklEQVRblfukJUuWwNvbG3l5eXByckJhYWGtjZ2lZsCAAUhLS0N2djZOnDiBH3/8EdOnT8e5c+fEjlaHnp4ebty4gQcPHuCvv/7CoEGDAACXLl2S9PxI4PHrRVJSEs6fPw8tLS0EBQUpRielKC8vD6tXr8aMGTOwYsUK/Pnnn1i9erXYsZ4pPj4eERERta6ytGvXDt99952IqZ4tMDAQ7dq1Q3h4OARBQGJiIoKDgyV9mfjjjz/G1KlTcefOHYSFhSEtLQ2zZs0SOxY1klqv2q5vUr9UC8nS0lJs2bIFJ0+ehFwuh7W1NWbPni3pN7Tz589j69atKC0thSAIkMvluHHjhmRHGRwcHBTzylTBsWPHYGZmhrZt22Lbtm3IzMxEr1694OvrC21tbbHjPdXff/+Nffv2ITExEUVFRfDx8cHEiRMl+QEjMzMTCxYsQElJCby9veHj44M9e/YgJiYG4eHhGDJkiNgR62VnZ4dvvvlG7BiNVt0m8euvv4aPjw9mzJgh2edwNVtbW+zcuRMbNmyAn58fjh8/jl9//RVr164VO1q9nJyccPDgwVrHVOF1788//0RmZiaqqqrQv39/WFhYcFRSRaj1iKSvry9KS0uRl5cHc3NzlJeXS3r13f79+zFlyhTMnz9f7CiNFhAQgOnTpyM5ORmenp749ttv0aVLF7Fj1atz5864dOmS5Ed6gccdbY4cOYKIiAj89ddf2L59OwIDA5GTk4PIyEjJXcb87rvvEBcXh4sXL2LEiBFYs2YNgoODJfvBDXg8epqeno7y8nLo6+sDALp27YrY2Fi8/fbb4oZrgJmZGaKjo9GjR49a221JbZpGzQ/0b7zxBlq3bo3s7GzFnD4pPz8MDQ3Rvn17WFhY4I8//sCkSZOwd+9esWM90zvvvINff/0VvXv3BvB4dF3qz+U9e/Zg4sSJij2eL126BHd3d+zfv1/kZNQYal1Injp1CkuXLkVVVRXi4+Nhb2+PtWvXwsbGRuxoT3Xr1i24ubmhY8eOcHR0xIgRIyQ9rwV4vGXR2LFjcf36dejr6yMyMlLSk6Rzc3Ph4uICIyMjaGtrQxAEya5oPHjwIOLj46Gnp4eoqCjY2trCzc0NgiDgww8/FDteHXPmzIGdnR3i4+MVHZpUYURBR0cHOjo6yMjIqDWvTOpvvgUFBcjMzERmZqbimNSnachkMkyYMEHsGI2mp6eHn3/+GRYWFkhLS4OVlRXKy8vFjvVUtra2kMlkePjwIVJTU9GxY0doaGggNze3Vsc0KTp8+DCqqqrg7u6OjRs3IiUlRaUGVJo7tS4k161bhz179mDmzJkwMjJCbGwsPvvsM8kWkv7+/vD398eZM2dw5MgRxMTEoEePHoiMjBQ7Wr1atGiBgoICvPPOO/jtt98wcOBAVFVViR2rXjExMWJHaDSZTKb4IJGZmYmJEycqjkvRoUOHkJSUhIkTJ8LU1BRjxoyR9HOhprVr1+Ls2bOws7ODXC7Hxo0bkZWVVWvxjdTs2rVL7AiNUj3iWN8G6lKUn58PExMTBAcHIyEhAf7+/khISICdnZ1kR1BV5fnwNP/617/g6+uLbdu2YdiwYTh8+DAbF6gQtS4k5XI5jIyMFLel2hqxJkEQUFFRgYqKCshkMsnPIZoyZQr8/PywefNmuLm5ISUlBd26dRM7Vr2MjIxw/PhxPHjwAMDj7YuuXbuGefPmiZysLk1NTRQVFaG0tBQ5OTkYPHgwAOD69euS3Kje3NwcixcvxoIFC/D9998jKSkJd+/ehZeXFyZNmoShQ4eKHbFe1Xmr/948PDwwduxYSReSqjY/WZU2UPfx8UFycjI6d+4MExMTaGhoYPPmzWLHeiZTU1MAjzf3VpXXuJr7yo4cORI5OTlo2bIljh07BuDxriAkfdJ7N3qB2rZti2PHjkEmk6GoqAixsbF48803xY5Vr5UrV+K7777Du+++C0dHRwQFBdVabS5FdnZ2GD16NGQyGRITE3H16lW8++67Yseq12effYbCwkLk5eWhb9++yMzMVMwlkhovLy84OzujsrIS48aNg7GxMY4cOYL169dj9uzZYserl5aWFoYPH47hw4fj3r17OHDgANauXSvpQvLVV1/FgwcPFG1UKyoqJL3IDVC9+cmqtIF6zTWoKSkpmDZtmohpno8qvcbVnJYBPN5rtKioSHGchaRqUOtCMjQ0FGFhYbh58yaGDx8Oa2trSX76rdahQwckJydLcnXrk5YsWfLM81JtOXj58mV8++23CAsLw9ixY/Hpp5/i008/FTvWU40ePRq9evXC/fv3FYuDWrVqhZUrV6pMGzQDAwNMmzZNsm/E1c9juVwOJycn2NraQlNTEydOnEDHjh1FTvdsqjY/WZU2UK85fUTVNjZRpde46veJ9evXw8/PT+Q01FRqXUgaGhpi3bp1YsdoUHx8PMaPH4/CwkLs2bOnznkpzsnp37+/2BGaxNDQEDKZDO+88w4uX74MZ2dnyXVbqcnExAQmJiaK21Ie1VNF1c/jJ5/PXbt2FSPOc1G1+cmqtoF6NanOSa6Pqr3GAY+3Ofv0009V7v81PabWheTIkSNrvbDKZDLo6uqiY8eO8Pf3V8wpEZuqfeIFHneIAYCSkhIcPHgQkyZNQn5+PuLi4uDl5SVyuvp17twZK1aswIQJE7BgwQLcvn1bJf//04thY2MDIyMj3LhxQ+woz23q1KkqNT/ZwcEB/fv3V4kN1K9cuaKYy5mfn6/4Xsq7PFRTxde4Nm3aYPTo0ejatWut6VxSvbJFtan1huQrV65Eu3btMG7cOACPV5VmZWXB1tYWsbGx+Oqrr8QN+ISdO3fC3t4ehoaGYkdpNB8fH1hYWMDPzw8lJSXYvn07cnNzJTcxvbpQqKqqQn5+Pt58802kp6fj1KlT8PPzU/SFpubF29sbW7duVWyd8iQpFgz5+fmIjIzElStX0LNnTyxYsABaWlq4evUqLC0tJXe5uPqKiyo1iLh+/fozz0tlEOJJhYWFqKqqQm5uLvr27Yv09HT8+OOPmDBhAszNzcWOV6/k5OSnHq8esCBpU+tC0sXFpc4T1NXVFUlJSU89J7aIiAjF/l+qso+ko6MjDh06VOvY0zoriK1moVD9lJfJZLh9+7aiJSU1T8eOHUOnTp3w1ltv4bvvvkNCQgK6dOmCTz75RJI9zadPnw5zc3MMGDAAqampAKQ9chMXFwcPDw+VKiRVUXZ2Nry8vLBq1SpFR6b169cjKSkJ27dvl3wThoKCApSVlUEQBMVK84EDB4odixpDUGOurq7CiRMnFLdPnDghjB8/Xrhz545gb28vYrJn++WXX4Tly5cLI0eOFBYuXCh2nGdydHQULl26pLj9559/Cq6uriImapySkhIhODhYeP/994Uff/xR7Dgkkh07dgiurq7ClStXhJycHKFHjx7Cvn37hOXLlwthYWFix3uqMWPGKL5/9OiR8OGHH4qYpmHOzs5iR2gWPv74Y+Hnn3+uc/zEiRPC5MmTX36g57Bp0yahd+/eQvfu3YWhQ4cKlpaWwrhx48SORY2k1nMkw8PDFfvaAcBbb72F1atXIz4+XrKrSAUV20dy8eLFmDZtmmJByP3797FmzRqRUz3bqVOnEBQUhMGDB+PQoUOS3+aFlOfAgQMq1T0IQK3XBG1tbcm/RtDLUVRU9NTdHN577z1ERUWJkKjxkpOTcfz4cYSFheGTTz5Bbm7uUxeekjSpdSFpbm6OpKQkFBYWQlNTU1EwSHUPPlXaR7LmPK2hQ4fCw8MDOjo66NixoyQvBwJAaWkpVq9ejR9//BErVqxQbPBNzZeqdQ96GqlnrblwpSZBBRauqJLKykrI5fI6c2TlcrnkV20bGxujdevW6Ny5My5duoSRI0di7dq1YseiRlLrQjI7OxtffPEFCgsLa61ak2ovWgMDA5XZRzIgIADm5uZwcHBAamoq9u7dK+l5WjVHIVNSUtCqVSuxI5EEqFr3IKBuYVa9qliqhVmHDh2wbds2sWOovX79+iE6Ohpz586tdfzzzz+X9Gp+AGjdujUOHDiArl27Yvfu3TA2NpZsT3OqS60X2zg4OGD8+PHo3LlzrU/tUt0D0c7ODt98843YMRrF3t4ehw8fBvC4C4izszO+/vprkVPVz9LSElpaWjA2Nq6z2bAU33zp5Th69CgiIyNRWVkJW1tbLFu2rFb3ICl21lC1FcXOzs61WuGRcpSUlMDLywu3bt2CpaUlWrRogezsbBgYGGDLli2Krk1SlJ+fj6+//hrTpk3D6tWrcfLkSXh7e2PMmDFiR6NGUOtC0s3NDfv37xc7RqPNmTMHFhYW6NGjB3R1dRXH+/XrJ2Kqp3ty1bvU3yxU7c2XXp78/Pxa3YOOHz8OXV1dlekeJHWhoaFYunSp2DGaBUEQ8PPPPyMnJwcaGhro1q0b+vbtK3asRqmoqEBubi60tLTw9ttvQ1NTU+xI1EhqXUhu3LgRBgYGsLGxqTXXUKr9tj09Pesck8lkkrwU/2QhKcXtlIiISPpOnz6NhQsXwtDQEHK5HKWlpVi7di2srKzEjkaNoNaFpK2tbZ1jvIz5YnTr1q1W6778/HyYmJjwUjERET0XV1dXhIeHw8LCAgCQlZWF5cuXIyEhQeRk1BjSnE3+gmRkZIgd4bl4eno+dQWmFEckqzdCJiIi+icEQVAUkQBgZWUl6b7xVJtaF5KFhYVYs2YN8vLysGnTJkRERGDJkiXQ19cXO9pTzZkzR/F9ZWUl0tPTJZuVcwqJiOif+OWXXwAAHTt2xNKlSzFu3DhoaWkhJSWFl7VViFpf2p47dy4GDx6M2NhYJCQkICYmBjk5OSq1FYWqLRgiIiJqjKetC6gm1fUBVJdaj0heu3YN48ePx969e6GjowM/Pz84OjqKHateN27cUHwvCAL+/PNPFBQUiJiIiIhIOXbt2iV2BHoB1LqQ1NTURHFxsWLe4dWrV+vs+i8lH330kSKrTCbDa6+9hqCgIJFTERERKc+ZM2ewc+dOFBYW1jrOEUnVoNaXtn/44QesXbsWN2/eRJ8+fXD+/HmsWrUKw4YNEztaHceOHUOnTp3w1ltv4bvvvkNCQgK6dOmCWbNmsZcuERGpreHDh8PX17fO1nxSbR5Ctal1IQkA9+7dw++//46qqir07NkThoaGYkeqY8eOHThy5AgiIiJQWVkJDw8PBAYGIicnB5qamggMDBQ7IhERkVJMmjQJsbGxYsegJlLrQjIvLw/nz5+Hvb09QkJCkJ2djeXLl0uu76ijoyPi4+Ohp6eHqKgo3LhxA+vWrYMgCPjwww9Vpm0iERHR8zp69CjS0tJgbW1dq8e9FFuUUl3SnTD4AixZsgRyuRwZGRm4evUqlixZgpUrV4odqw6ZTAY9PT0AQGZmJt577z3FcSIiInWWmJiI27dv4+zZs8jMzFR8kWpQ68U2Dx8+hLOzMwIDA+Hg4IC+ffvi0aNHYseqQ1NTE0VFRSgtLUVOTg4GDx4M4HF/6JqfzoiIiNTN3bt32WJXhan1iKSmpiZSU1Px/fffY9iwYUhLS5Pkqm0vLy84OzvD3d0d48aNg7GxMY4cOYIpU6Zg+vTpYscjIiJSmu7du+PYsWPsZqOi1HqO5OXLl/HVV19h2LBhGDVqFPz8/ODt7Q1LS0uxo9WRn5+P+/fvK7IdP34curq6GDBggMjJiIiIlMfGxgZ3795VTOcSBAEymQw5OTkiJ6PGUOtCEgBu374NY2NjnDlzBpcvX8bYsWOhq6srdiwiIqJmbc+ePZg4cSIA4I8//oC5ubni3MqVK7mPsoqQ3nXeFygkJAQbNmzAn3/+ifnz5+PixYt8YhIREUlAzfa//v7+tc6dPXv2ZcehJlLrQjIrKwthYWH45ptvMG7cOKxatQr/+c9/xI5FRETU7NW8IPrkxVE1v1iqVtS6kKyqqoJcLkd6ejqGDBmCsrIylJWViR2LiIiIanhyuztuf6c61LqQdHZ2ho2NDUxNTdGjRw+MHTsW7u7uYsciIiJq9lgsqge1X2wjl8sVW/7cu3cPBgYGIiciIiKibt26wcTEBMDjnUuqvxcEAXfu3EFWVpaY8aiR1LqQPH/+PLZu3YrS0lIIggC5XI4bN24gIyND7GhERETN2vXr15953tTU9CUloX9CrdumBAQEYPr06UhOToanpye+/fZbdOnSRexYREREzR4LRfWg1oWkjo4Oxo4di+vXr0NfXx+RkZFwcHAQOxYRERGRWlDrxTYtWrRAQUEB3nnnHfz222/Q1NRkCyYiIiKiF0StC8mpU6fCz88P77//Pg4ePIgxY8agW7duYsciIiIiUgtqeWk7Pz8fkZGRuHLlCnr27Am5XI7ExERcvXpVkn22iYiIiFSRWq7anj59OszNzTFgwACkpqYCAMLDw0VORURERKRe1HZEcseOHQCAwYMHw9nZWeREREREROpHLedIamtr1/q+5m0iIiIiejHUspB8EtswEREREb14ajlHsmbbJeB/rZcEQYBMJkN6erqI6YiIiIjUg1oWkmy7RERERKR8allIEhEREZHyNYs5kkRERET04rGQJCIiIqImYSFJRERERE3CQpKIiIiImoSFJBERERE1yf8BG1mL/WoVbBAAAAAASUVORK5CYII=
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="We-can-see-that-about-20%-of-Age-and-majority-of-Cabin-calues-are-null.-2-null-values--for-Embarked-column-also.">We can see that about 20% of Age and majority of Cabin calues are null. 2 null values  for Embarked column also.<a class="anchor-link" href="#We-can-see-that-about-20%-of-Age-and-majority-of-Cabin-calues-are-null.-2-null-values--for-Embarked-column-also.">&#182;</a></h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[18]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Drop Cabin column</span>
<span class="n">titanic_df</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="s1">&#39;Cabin&#39;</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[20]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Drop nulls from the dataframe</span>
<span class="n">titanic_df</span><span class="o">.</span><span class="n">dropna</span><span class="p">(</span><span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[21]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Verifying that we dont have any more nulls. </span>
<span class="c1"># Notice we dont see any black bars on the heatmap so all nulls have been dropped. - </span>

<span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">titanic_df</span><span class="o">.</span><span class="n">isnull</span><span class="p">(),</span><span class="n">cmap</span><span class="o">=</span><span class="s2">&quot;YlGnBu&quot;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[21]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x127870f28&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Data-Analysis">Data Analysis<a class="anchor-link" href="#Data-Analysis">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="How-many-survivals-for-Male-and-Female">How many survivals for Male and Female<a class="anchor-link" href="#How-many-survivals-for-Male-and-Female">&#182;</a></h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[22]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sns</span><span class="o">.</span><span class="n">set_style</span><span class="p">(</span><span class="s2">&quot;whitegrid&quot;</span><span class="p">)</span> 
<span class="n">sns</span><span class="o">.</span><span class="n">countplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="s2">&quot;Survived&quot;</span><span class="p">,</span> <span class="n">hue</span><span class="o">=</span><span class="s2">&quot;Sex&quot;</span><span class="p">,</span> <span class="n">data</span><span class="o">=</span><span class="n">titanic_df</span><span class="p">,</span> <span class="n">palette</span><span class="o">=</span><span class="s2">&quot;Set3&quot;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[22]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x12795f278&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAtgAAAHlCAYAAADP34vrAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjAsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+17YcXAAAgAElEQVR4nO3dfVTUdd7/8dcgAYOFWiSlpiJbqJWKYGiFWqkZaqFm283SpqWiJqJXd+p2aW6lrXriksqSLLNtr6MVeV9bZLteqWlMprmLekVgeUN4E67CMDDM/P7ot3Mtqeugn+E76PNxzp6z872Z73vmcPDZ93yYsXm9Xq8AAAAAGBFi9QAAAADA+YTABgAAAAwisAEAAACDCGwAAADAIAIbAAAAMIjABgAAAAwKtXoA0xwOh9UjAAAA4AKRmJh40rbzLrClU79QAAAAwKTT3dhliQgAAABgEIENAAAAGERgAwAAAAYR2AAAAIBB5+UfOQIAAEDyer0qLy9XeXm51aM0Ws2bN1fz5s1ls9n8PofABgAAOE+VlpbKZrOpXbt2Cglh4UJ9eTwelZWVqbS0VFdeeaXf5/FOAwAAnKecTqdiYmKI67MUEhKimJgYOZ3O+p0XoHkAAABgMZvNVq+lDTjZ2byHBDYAAABgEGuwAQAAEDAFBQWaP3++KioqVFtbq27dumnatGlq2rSp1aMFDHewAQAAEBDV1dWaOHGinn/+ea1atUqrVq2SJM2dO9fiyQKLwAYAAEBAOJ1OnThxQsePH5ckNWnSRFlZWbrzzjtVXl6urKwsDRs2THfddZfy8vIkSfPnz9e4ceMkSV9//bX69u2ro0ePWvYazgZLRAAAABAQzZo108SJE3X//ferVatWuuGGG3Tbbbfplltu0RNPPKEBAwYoNTVVJ06c0H333adrr71WEydO1D333KP3339fixYt0uzZs3XppZda/VLqhcAGAABAwIwZM0YjRozQpk2btGXLFj311FMaMGCANmzYoMLCQr322muSpIqKCu3evVvx8fGaO3euhg4dqt/85jfq1auXxa+g/ghsAAAABMRXX32lb775Rr/97W81aNAgDRo0SCNHjtTgwYMVGRmp3NxcXXHFFZKkw4cPKyoqSpK0d+9eNWvWTH/729/k8Xga3ed4N65pAQAA0Gg0a9ZMr7zyiv72t7/5tn377beKj49Xr169tHTpUknSoUOHdOedd+q7777TkSNHNGvWLC1evFhNmzbVG2+8YdX4Z4072AAAAAiIuLg4zZ07VzNnztRPP/2k0NBQxcbGKicnR2FhYZo5c6aGDBkit9utrKwsdezYURkZGRoxYoQ6duyomTNnKi0tTTfffLM6duxo9cvxm83r9XqtHsIkh8OhxMREq8cAAACwXHFxsWJjY60eo9E73ft4uu7kDnaAVLlrVOvxWD0GgkyTkBBFhF5k9RgAACCACOwAqfV4tHT7FqvHQJB5sGuy1SMAAIAA448cAQAAAIMIbAAAAMAgAhsAAAAwiMAGAAAADCKwAQAAAIP4FBEAAIALSEN8lHBDfSztli1b9Lvf/U6ffPJJwK9VHw0S2O+++64WLlyon376Sdddd51mzpypuLg4paena8eOHb7vl09JSdGCBQskSS+99JLeeecdeTwePfDAA8rMzGyIUQEAAM5rDfFRwhf6x9IGfInIrl27NG/ePC1atEgOh0NJSUmaOXOmJGnPnj366KOPtG3bNm3bts0X1x9//LFWr16tFStWKC8vT6tXr9aGDRsCPSoAAAAa0JYtW3Tfffdp+vTpSkhI0NChQ7Vt2zbdf//9SkhIUGZmpjwejz777DMNGzZMSUlJuvnmm/XHP/7xlM/3wQcfaMCAAerZs6emTp2qioqKBn5FPwt4YHfs2FHr16/Xr371K5WXl+vEiRNq0aKFSktL5fV6deWVV550zpo1a3TfffcpJiZGrVu3Vnp6uvLy8gI9KgAAABrYV199pR49eqigoECXXnqpxowZo2eeeUaffPKJvvrqK/3lL3/R448/rqeffloFBQWaM2eO5syZoxMnTtR5noKCAs2bN085OTn67LPP5Ha79cILL1jymhrkjxybNm2qTZs26cYbb9SKFSs0fvx47d69W+Hh4br33nvVq1cvTZw4UYcOHZIklZSUqEOHDr7z27dvr6KiooYYFQAAAA2oWbNmSktLU5MmTZSQkKAbbrhBV199taKjo3XNNdfoyJEjWrFihRISEvTTTz8pJCREbrdbx44dq/M8K1eu1K9//WvFx8fLbrcrKytLK1eulNfrbfDX1GB/5JiUlKQdO3ZoyZIlysjI0BNPPKEuXbroqaee0qWXXqrnn39ejz/+uJYsWSKn0ym73e471263q6qqyu9r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AAAMIrABAAAAgwhsAAAAwCACGwAAADCIwAYAAAAMIrABAAAAgwhsAAAAwCACGwAAADCIwAYAAAAMIrABAAAAgwhsAAAAwCACGwAAADCIwAYAAAAMIrABAAAAgxoksN99913deuutSkhIUHp6uoqKiiRJy5YtU0pKihITEzVjxgzV1tb6znnppZfUq1cvJScna8GCBQ0xJgAAAHDOAh7Yu3bt0rx587Ro0SI5HA4lJSVp5syZ2rlzp7Kzs7VkyRLl5+ersLBQy5cvlyR9/PHHWr16tVasWKG8vDytXr1aGzZsCPSoAAAAwDkLeGB37NhR69ev169+9SuVl5frxIkTatGihdauXavBgwcrLi5OLVq00NixY5WXlydJWrNmje677z7FxMSodevWSk9P9+0DAAAAglmDLBFp2rSpNm3apBtvvFErVqzQ+PHjVVJSog4dOviOad++vW/pyL/bBwAAAASz0Ia6UFJSknbs2KElS5YoIyNDbdu2ld1u9+232+2qqqqSJDmdztPu80dhYaG5wc9Sq3ZtrR4BQcjtdquw6DurxwAAAAHUYIEdFhYmSRo9erQWL16syMhIuVwu336n06nIyEhJUkRExGn3+aNTp06Gpj57FdWuMx+EC05oaGhQ/HwCAIBz53A4Trk94EtE8vPzNWnSJN9jj8ejmpoahYWFqaSkxLf9X5eFxMbGau/evafcBwAAAASzgAf2tddeq88//1ybNm1STU2NFixYoGuuuUaPPPKIVq1apT179qi8vFy5ublKTU2VJKWmpurtt9/WwYMHdeDAAb3zzjsaNGhQoEcFAAAAzlnAl4hceeWVevHFF/Xss8/q0KFDSkpK0oIFC9SyZUtlZWUpIyNDx48f15AhQ5Seni5JGjhwoIqLi3XPPfeopqZG6enp6tevX6BHBQAAAM6Zzev1eq0ewiSHw6HExESrx1BFtUtLt2+xegwEmQe7JqtpWLjVYwAAAANO1518VToAAABgEIENAAAAGERgAwAAAAYR2AAAAIBBBDYAAABgEIENAAAAGERgAwAAAAYR2AAAAIBBBDYAAABgEIENAAAAGERgAwAAAAYR2AAAAIBBBDYAAABgEIENAAAAGERgAwAAAAYR2AAAAIBBBDYAAABgEIENAAAAGERgAwAAAAYR2AAAAIBBBDYAAABgEIENAAAAGERgAwAAAAYR2AAA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5HQ6LZoIAIJfy5YtrR4BOGsENhBgdrtd1dXVdbZVVVXxzaQAAJynCGwgwGJjY1VSUuJ7fPz4cR07dkxt27a1bigAABAwBDYQYD179tTBgwe1bt06VVdXKzs7W7179+YONgAA5ykCGwiwiIgILVy4UIsWLVJycrJKSko0a9Ysq8cCAAABYvN6vV6rhwAAAADOF9zBBgAAAAwisAEAAACDCGwAAADAIAIbAAAAMIjABgAAAAwisAEAAACDCGwACHJHjhzR1KlTdeONN6pbt2669dZbNWfOHDmdTqPXWbVqlYYNG2b0OSXpqaee0n/+538af14ACFYENgAEucmTJ8vj8WjdunX6+uuvtWTJEjkcDv3ud78zep0777xTeXl5Rp8TAC5EBDYABLnt27erf//+at68uSSpbdu2mjZtmqKjo7Vv3z7Fx8fr0KFDvuMffvhh5eTkSPr57nFWVpb69++vPn36KDMz86S7yampqfrggw+Ul5engQMHyuPx6JZbbtGaNWt8x5SVlenaa6/Vvn375PF4lJubq/79++uGG27Q2LFjtX//ft+xX375pdLS0tStWzc98sgj+umnnwL59gBA0CGwASDIpaamavr06XruueeUn5+vo0ePKiEhQVOnTvXr/M2bN+uNN97QmjVrdN999+mjjz5SdXW1JGnnzp06ePCgbr/9dt/xISEhSktL06pVq3zb1qxZo8TERLVp00ZLly7Vu+++q1dffVWff/65rrvuOo0ePVput1tHjx5VRkaG7rnnHhUUFOg3v/mN/ud//sfsGwIAQY7ABoAg99xzz2nq1Knau3evnnjiCfXq1Uv33nuvtm/f7tf53bt311VXXaVLLrlEPXv21MUXX6y//vWvkqSVK1dq4MCBioyMrHPOsGHDtGnTJh09etR33D/XZy9btkzjxo1TXFycwsLC9Oijj+qnn37S1q1b9dlnn+myyy7T/fffr9DQUPXt21d9+vQx+G4AQPAjsAEgyP3zjvKiRYtUUFCgDz74QK1bt9aoUaP0j3/844znX3755b7/b7PZNHToUK1evVput1tr167V8OHDTzrnqquuUkJCgtatW6fdu3frhx9+0IABAyRJBw4c0KxZs5SUlKSkpCT16NFDlZWV2r9/vw4dOqSYmJg6z9WmTZtzfAcAoHEhsAEgiG3YsEFJSUmqqKiQ9HNsd+7cWbNnz1ZFRYUqKyslSbW1tb5zysvL/+1zDh06VH/961/16aefqmnTpkpKSjrlccOGDdO6deu0Zs2aOne5Y2JiNHfuXBUUFPj+l5eXpyFDhigmJkYHDhyo8zxlZWVn/foBoDEisAEgiPXo0UNRUVGaPn26fvjhB0nS0aNH9fLLL6tNmzbq0qWLLrnkEq1atUput1uffvqpdu3a9W+fs02bNuratavmzJmjoUOHnva422+/XXv27KmzPESShg8frldeeUU//PCDvF6v8vLydNddd+nHH3/ULbfcooqKCi1evFg1NTXauHGjPv30UzNvBgA0EgQ2AAQxu92ud955R3a7XQ888IC6deum1NRU7d+/X0uXLlVYWJhmzpyp999/Xz169NDKlSs1aNCgMz7vsGHDVFpaqrS0tNMeExkZqQEDBigiIqLOXe6HH35Y/fv318iRI9W9e3ctXbpUL7/8stq1a6fmzZvr9ddf10cffaQePXooJydH/fr1M/JeAEBjYfN6vV6rhwAAAADOF9zBBgAAAAwisAEAAACDCGwAAADAIAIbAAAAMIjABgAAAAwisAEAAACDCGwAAADAIAIbAAAAMIjABgAAAAz6f/eXeGv1QS5IAAAAAElFTkSuQmCC
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Most-passengers-who-did-not-survive-were-Male">Most passengers who did not survive were Male<a class="anchor-link" href="#Most-passengers-who-did-not-survive-were-Male">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="How-many-survivals-by-passenger-class?">How many survivals by passenger class?<a class="anchor-link" href="#How-many-survivals-by-passenger-class?">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[23]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sns</span><span class="o">.</span><span class="n">countplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="s2">&quot;Survived&quot;</span><span class="p">,</span> <span class="n">hue</span><span class="o">=</span><span class="s2">&quot;Pclass&quot;</span><span class="p">,</span> <span class="n">data</span><span class="o">=</span><span class="n">titanic_df</span><span class="p">,</span> <span class="n">palette</span><span class="o">=</span><span class="s2">&quot;Set3&quot;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[23]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x127b8e3c8&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAtgAAAHlCAYAAADP34vrAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjAsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+17YcXAAAgAElEQVR4nO3de7SWdZ3//9cGFDYaA0lqoyIbKg+VysHBcvBQHlEKPJSNMSstFU8IVmPqFObYaMsaGcmo0FKcmoU5pA4emtApU0ljL7QcUQvZpCii0nYUNvv8+2N+s79Dam7lc3NvNo/HWqzlfR3f973kXk+udd33XdPZ2dkZAACgiD7VHgAAAHoTgQ0AAAUJbAAAKEhgAwBAQQIbAAAKEtgAAFBQv2oPUFp9fX21RwAAYCsxZsyY1yzrdYGdvP4TBQCAkt7owq5bRAAAoCCBDQAABQlsAAAoSGADAEBBvfJDjgAAVE9nZ2caGxvT2NhY7VE22eDBgzN48ODU1NR0ex+BDQBAUatXr05NTU1233339Omz5d4w0dHRkTVr1mT16tV597vf3e39ttxnDABAj9TU1JSddtppi47rJOnTp0922mmnNDU1vbX9KjQPAABbqZqamrd0S0VP9naei8AGAICC3IMNAEBVPPPMMznyyCPznve8JzU1Neno6Ej//v1zwQUXZOzYsa+7z0c+8pHMnTs3I0eO3MzTdp/ABgCgagYPHpxbb7216/HPf/7zTJ8+Pffdd18Vp9o0AhsAgB7jgAMOyAsvvJBVq1bln/7pn/Jf//Vf6devX0477bR8/OMf79quvb09//AP/5Bly5alsbEx73znOzNr1qzstNNOufHGG3PzzTenT58+2W233fL1r38922yzTb7yla/kscceS01NTQ455JCcd955FXkO7sEGAKDHuPnmm1NXV5frr78+22+/fe66667ccMMN+c53vpN169Z1bffwww+npaUl8+fPz09/+tPstttuWbBgQdrb2zN79uzcdNNN+clPfpK6urr87ne/y9KlS/Pcc8/llltuyb/+67/mD3/4w0bHK8kVbAAAqqaxsbHrynRra2t22223XH311fn85z+fyy67LEmyww475M4779xovzFjxmTQoEH50Y9+lJUrV+bhhx/OTjvtlL59+2bcuHGZPHlyPvrRj+bQQw/NPvvsk8bGxqxatSpTpkzJQQcdlOnTp2e77baryHMS2AAAVM2f3oP9v/r27bvR1+OtXLkyO++8c9fje+65J9/4xjdyyimn5Oijj07fvn3T2dmZJJk9e3Z++9vf5pe//GW+8IUv5KyzzsoJJ5yQhQsX5sEHH8wvf/nLfOITn8i8efPy3ve+t/hzcosIAAA9zrhx43LbbbclSdauXZuTTz45r7zyStf6Bx54IEcccUROPPHE7L777vnFL36R9vb2rF27NkcccUR23333nHXWWTn++OPz2GOP5Z577sl5552XAw88MBdddFHe9773Zfny5RWZ3RVsAAB6nHPPPTeXXHJJJk6cmJqamlx88cUZOnRo1/pPfvKT+fznP5+77747NTU12WefffLMM8/kne98Zz796U/nk5/8ZGprazNo0KD84z/+Y3bcccf87Gc/yzHHHJMBAwZkzz33zEc+8pGKzF7T+b/X0nuJ+vr6jBkzptpjAABstVasWJG6urpqj1HMGz2fN+pOV7CBN9Ta2paOjl71b/Aep0+fmmyzjbdigN7Euzrwhjo6OvPgA7+v9hi92rgPv6faIwBQmA85AgBAQQIbAAAKEtgAAFCQwAYAgIIENgAAFORbRAAAqLgNba1p7+io6Dn69umTAf22eUv7zJ07Nw0NDfna175WbA6BDQBAxbV3dGTeIw9W9Bx/u++4bm/b0tKSa665Jt/97ndz/PHHF51DYAMAsNWZOXNmXnzxxZx00klpbW0temz3YAMAsNWZMWNG5s6dmx122KH4sQU2AABbnR133LFixxbYAABQkMAGAICCBDYAABQksAEAoCBf0wcAQMX17dPnLX1P9ds9x1t17rnnFp9DYAMAUHFv9RcWt2RuEQEAgIIENgAAFCSwAQCgIIENAAAFCWwAAChIYAMAQEECGwAACvI92AAAVFxnZ2s6Ozsqeo6amj6pqan+920LbAAAKq6zsyPNzfdV9Bz9+/91amq6t+29996bK6+8MqtWrcqwYcNy8cUXZ//99y8yx2a5ReTee+/NxIkTM3r06EyaNCm//vWvkyRf+tKX8sEPfjCjRo3KqFGjcsIJJ3TtM3/+/IwfPz5jxozJzJkz097evjlGBQCgl1u7dm3OP//8fOELX8iSJUty6qmn5pxzzsn69euLHL/igf3nnsATTzyRH/zgB1m6dGmWLl2am2++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WbZsWW666abNMSoAAGySzRLYM2fOzGOPPZaTTjqpa1l7e3uWL1+ePfbY4zXb33777Tn22GMzcuTIDBkyJGeccUYWLFiwOUYFAIBN0q3A/sxnPvO6y4877rhunWTGjBmZO3dudthhh65lK1asSN++fTN9+vQccMABOeWUU7JixYokSUNDQ0aMGNG17fDhw7N8+fJunQsAAKqp3xuteOaZZ3LttdcmSZYsWZJLLrlko/WvvPJKnn322W6dZMcdd3zNsldffTWjR4/OF7/4xdTV1eW73/1uzjzzzCxcuDBNTU2pra3t2ra2tjYbNmzo1rmSZNmyZd3eFnhjw4YNr/YIvV5bW1ueeur31R4DgILeMLB33XXX1NbWprGxMUnS3Ny80frtttsuV1111ds+8X777Zfrrruu6/G5556bG264IQ0NDRkwYMBG52tqasrAgQO7fey99trrbc8F/D/Nza3VHqHX69evn/csgC1UfX396y5/w8BOkgsuuCBJMnLkyHzuc58rOtCvfvWrrF69OpMmTUqSdHR0pL29Pdtuu23q6urS0NDQte2f3jICAAA91Z8N7P/1uc99Lk8++WRWrFjxmq/LmzBhwts++eWXX5699tordXV1mTVrVvbYY48MGzYsRx99dM4888xMnjw5O+64Y+bOnbtJ5wEAgM2lW4H9z//8z/nOd76Td73rXenX7//tUlNT87bD94ADDsi0adMyderUNDY2ZvTo0Zk1a1aSZJ999sn06dMzderUvPLKK5k4cWKmTJnyts4DAACbU01nZ2fnm2108MEH57LLLsv48eM3x0ybpL6+PmPGjKn2GNArNDe35sEHfACvksZ9+D3p33+bao8BwNvwRt3Zra/pa2pqyoEHHlh8KAAA6G26FdhHHXVUbrzxxkrPAgAAW7xu3YO9cuXK3HTTTfn2t7+90Y/FJMkdd9xRkcEAAGBL1K3AnjRpUtfX6QEAAG+sW4E9efLkSs8BAAC9QrcCe8qUKampqXnddfPmzSs6EAAAbMm6Fdh/+vV8jY2NufPOO3PCCSdUZCgAANhSdSuwTz/99NcsO/7443PRRRfl7LPPLj4UAABsqbr1NX2v5y//8i/z+9/7AQoAAPi/unUF+0+/iq+9vT0//elP8973vrciQwEAwJaqW4H9jW98Y6PHffv2zfDhw3PJJZdUYiYAANhidSuw77nnnkrPAQAAvUK3AjtJHn744SxYsCDPPfdchg4dmkmTJmXcuHGVnA0AALY43fqQ46JFi/KZz3wmra2t2XfffdPR0ZHTTz89d955Z6XnAwCALUq3rmBfc801ufrqq3PQQQd1LTvmmGNy5ZVX5uijj67YcAAAsKXp1hXsp59++jU/NvPXf/3XWbVqVUWGAgCALVW3AnuXXXbJ/fffv9GyBx54ILvuumtFhgIAgC1Vt24ROeuss3L22WdnwoQJ2WWXXbJq1arceeed+frXv17p+QAAYIvSrcA+8sgj84c//CH19fVZs2ZN2tracu2112bs2LGVng8AALYo3bpF5KabbsqcOXMyY8aMXHfddTn22GNz1llnZeHChZWeDwAAtijdCuzvfe97ueGGG7LHHnskSU488cRcd911mT17dkWHAwCALU23Anvt2rXZe++9N1q2995756WXXqrIUAAAsKXqVmC/733vy49+9KONls2fP7/rijYAAPA/uvUhxwsvvDCnnXZa/uVf/iU777xzVq9enZdffjnXXnttpecDAIAtSrcCe999981//Md/5Oc//3nWrFmTnXfeOQcffHD+4i/+otLzAQDAFqVbgZ0kgwcPzqRJkyo5CwAAbPG6dQ82AADQPQIbAAAKEtgAAFCQwAYAgIIENgAAFCSwAQCgIIENAAAFCWwAAChIYAMAQEECGwAAChLYAABQkMAGAICCBDYAABQksAEAoKB+1R4AAJJkQ1tr2js6qj1Gr9e3T58M6LdNtceAXk1gA9AjtHd0ZN4jD1Z7jF7vb/cdV+0RoNdziwgAABQksAEAoCCBDQAABQlsAAAoSGADAEBBAhsAAAoS2AAAUJDABgCAggQ2AAAUJLABAKAggQ0AAAUJbAAAKEhgAwBAQQIbAAAKEtgAAFCQwAYAgIIENgAAFCSwAQCgIIENAAAFCWwAAChIYAMAQEECGwAAChLYAABQkMAGAICCBDYAABS0WQN77ty5ufjii7seL1q0KIcddlhGjRqVadOmZd26dV3r5s+fn/Hjx2fMmDGZOXNm2tvbN+eoAADwtmyWwG5paclVV12Vb37zm13Lnn/++VxwwQW54oorcv/996etrS1z5sxJkjz66KOZNWtWrr/++ixatCjLli3LTTfdtDlGBQCATbJZAnvmzJl57LHHctJJJ3Ut+9nPfpZx48Zl7NixGThwYKZNm5YFCxYkSW6//fYce+yxGTlyZIYMGZIzzjijax0AAPRkmyWwZ8yYkblz52aHHXboWtbQ0JARI0Z0Pa6rq8tLL72UxsbG16wbPnx4li9fvjlGBQCATdJvc5xkxx13fM2ypqamDB48uOtx//79U1NTkw0bNqSpqSm1tbVd62pra7Nhw4Zun2/ZsmWbNjCQJBk2bHi1R+j12tra8tRTv6/2GD3CX+4+rNojbBXa2tqybPlT1R4DerXNEtivp7a2Ni0tLV2Pm5ub09nZmYEDB2bAgAFpbm7uWtfU1JSBAwd2+9h77bVX0Vlha9Xc3FrtEXq9fv36ec/6/61raX7zjdhk/p+Dcurr6193edW+pq+uri4NDQ1dj1esWJGhQ4dm0KBBr1n3p7eMAABAT1W1wP7oRz+axYsX56GHHsr69evzrW99KxMmTEiSHH300bntttvy5JNPprGxMXPnzu1aBwAAPVnVbhHZeeedc+WVV+bLX/5yXnzxxYwfPz7nn39+kmSfffbJ9OnTM3Xq1LzyyiuZOHFipkyZUq1RAQCg2zZrYJ977rkbPT7kkENyyCGHvO62J554Yk488cTNMBUAAJTjp9IBAKAggQ0AAAUJbAAAKEhgAwBAQQIbAAAKEtgAAFCQwAYAgIIENgAAFCSwAQCgoKr9VDpsqs7O1nR2dlR7jF7Ov8EB4K0S2GyxOjs70tx8X7XH6NX69Blf7REAYIvj8hQAABQksAEAoCCBDQAABQlsAAAoSGADAEBBAhsAAAryNX0AAAW1tralo6Oz2mP0an361GSbbXpuxvbcyQAAtkAdHZ158IHfV3uMXm3ch99T7RH+LLeIAABAQa5gA8BWpH+/pKOjudpj9HKuX27tBDYAbEVq0pnm5vurPUav1qfP+GqPQJX5JxYAABQksAEAoCCBDQAABQlsAAAoSGADAEBBAhsAAAoS2AAAUJDABgCAggQ2AAAUJLABAKAggQ0AAAUJbAAAKEhgAwBAQQIbAAAKEtgAAFCQwAYAgIIENgAAFCSwAQCgIIENAAAFCWwAAChIYAMAQEECGwAAChLYAABQkMAGAICCBDYAABQksAEAoCCBDQAABfWr9gC91Ya21rR3dFR7jF6ttl9ntUcAAHgNgV0h7R0dmffIg9Ueo1c7fcxfVXsEAIDXcIsIAAAUJLABAKAggQ0AAAUJbAAAKEhgAwBAQQIbAAAKEtgAAFCQwAYAgIIENgAAFCSwAQCgIIENAAAFCWwAAChIYAMAQEECGwAAChLYAABQkMAGAICCqh7Ys2fPzgc+8IGMGjUqo0aNyoEHHpgkWbRoUQ477LCMGjUq06ZNy7p166o8KQAAvLmqB/YTTzyRyy67LEuXLs3SpUtz//335/nnn88FF1yQK664Ivfff3/a2toyZ86cao8KAABvqkcE9p577rnRsp/97GcZN25cxo4dm4EDB2batGlZsGBBlSYEAIDuq2pgr1+/Ps8880xmzZqVD33oQ/nkJz+Z3/zmN2loaMiIESO6tqurq8tLL72UxsbGKk4LAABvrl81T/7SSy9l7Nix+exnP5t99903t956a84444wceuihGTx4cNd2/fv3T01NTTZs2NCt4y5btqxSI3fbX+4+rNoj9H6dndWeoNfzEldeW1tbnnrq99Ueo0fwvrmZ+ItdcV7iyuvp751VDezddtstN954Y9fjE088MfPmzUt9fX2GDh3atby5uTmdnZ0ZOHBgt4671157FZ/1rVrX0lztEXq/mppqT9DreYkrr1+/fj3iPasn8L65mfiLXXFe4srrKe+d9fX1r7u8qreILFu2LDfccMNGy1paWvK3f/u3aWho6Fq2YsWKDB06NIMGDdrMEwIAwFtT1cAeMGBAZs2alcWLF6e9vT033nhjWltbc8ghh2Tx4sV56KGHsn79+nzrW9/KhAkTqjkqAAB0S1VvEamrq8sVV1yRr371q1m9enX23HPPzJkzJ7vsskuuvPLKfPnLX86LL76Y8ePH5/zzz6/mqAAA0C1VDewkOfLII3PkkUe+ZvkhhxySQw45ZPMPBAAAm6Dq34MNAAC9icAGAICCBDYAABQksAEAoCCBDQAABQlsAAAoSGADAEBBAhsAAAoS2AAAUJDABgCAggQ2AAAUJLABAKAggQ0AAAUJbAAAKEhgAwBAQQIbAAAKEtgAAFCQwAYAgIIENgAAFCSwAQCgIIENAAAFCWwAAChIYAMAQEECGwAAChLYAABQkMAGAICCBDYAABQksAEAoCCBDQAABQlsAAAoSGADAEBBAhsAAAoS2AAAUJDABgCAggQ2AAAUJLABAKAggQ0AAAUJbAAAKEhgAwBAQQIbAAAKEtgAAFCQwAYAgIIENgAAFCSwAQCgIIENAAAFCWwAAChIYAMAQEECGwAAChLYAABQkMAGAICCBDYAABQksAEAoCCBDQAABQlsAAAoSGADAEBBAhsAAAoS2AAAUJDABgCAggQ2AAAUJLABAKAggQ0AAAUJbAAAKEhgAwBAQQIbAAAKEtgAAFCQwAYAgIIENgAAFNRjA7u+vj4TJ07Mfvvtl1NOOSVr1qyp9kgAAPCmemRgb9iwIdOmTcvZZ5+dhx56KMOHD8/ll19e7bEAAOBN9av2AK9n8eLFGTp0aI466qgkyYwZM3LggQfm1Vdfzfbbb1/l6QAA4I31yCvYK1euzIgRI7oeDxo0KIMGDcrKlSurONLE41sAAAdDSURBVBUAALy5HnkFe/369RkwYMBGywYMGJCmpqZu7V9fX1+Jsd6yv+qzXbVH6NUeXvpfSQZXe4xe7rfZblC1Z+jdHn30N9UeoUfxvll53js3B++dldbT3zt7ZGDX1tampaVlo2UbNmzIdtu9+RvvmDFjKjUWAAC8qR55i0hdXV0aGhq6Hr/yyit5+eWXM2zYsOoNBQAA3dAjA/uAAw7Ic889lzvuuCMtLS2ZNWtWDjrooG5dwQYAgGqq6ezs7Kz2EK/nkUceycyZM7Ny5cqMHj06X//61zN06NBqjwUAAH9Wjw1sAADYEvXIW0QAAGBLJbABAKAggQ2bQX19fSZOnJj99tsvp5xyStasWVPtkQC2CHPnzs3FF19c7THgLRHYUGEbNmzItGnTcvbZZ+ehhx7K8OHDc/nll1d7LIAeraWlJVdddVW++c1vVnsUeMt65A/NQG+yePHiDB06NEcddVSSZMaMGTnwwAPz6quvZvvtt6/ydAA908yZM/Piiy/mpJNOSmtra7XHgbfEFWyosJUrV2bEiBFdjwcNGpRBgwZl5cqVVZwKoGebMWNG5s6dmx122KHao8BbJrChwtavX58BAwZstGzAgAFpamqq0kQAPd+OO+5Y7RHgbRPYUGG1tbVpaWnZaNmGDRv8MikA9FICGyqsrq4uDQ0NXY9feeWVvPzyyxk2bFj1hgIAKkZgQ4UdcMABee6553LHHXekpaUls2bNykEHHeQKNgD0UgIbKmzAgAGZM2dOvve972XcuHFpaGjIpZdeWu2xAIAKqens7Oys9hAAANBbuIINAAAFCWwAAChIYAMAQEECGwAAChLYAABQkMAGAICCBDZAD/fSSy/lwgsvzIc//OHst99++chHPpIrrrgiTU1NRc9z22235bjjjit6zCT50pe+lK985SvFjwvQUwlsgB5uxowZ6ejoyB133JGHH344119/ferr6/P3f//3Rc/zsY99LAsWLCh6TICtkcAG6OEeeeSRHH744Rk8eHCSZNiwYbnooosydOjQPPPMM9ljjz3ywgsvdG3/2c9+NrNnz07yP1ePp0+fnsMPPzwHH3xwpk2b9pqryRMmTMhPfvKTLFiwIEcddVQ6Ojpy6KGHZuHChV3brFmzJu9///vzzDPPpKOjI3Pnzs3hhx+ev/qrv8oZZ5yRVatWdW3761//OpMmTcp+++2Xz33uc/njH/9YyZcHoMcR2AA93IQJE3LxxRfna1/7WhYtWpS1a9dm1KhRufDCC7u1/+LFi/P9738/CxcuzKc+9ancddddaWlpSZI8+uijee6553LkkUd2bd+nT59MmjQpt912W9eyhQsXZsyYMdl1110zb968/PjHP853vvOd3HffffnABz6Q0047LW1tbVm7dm2mTp2aT3ziE1myZEk+/elP55e//GXZFwSghxPYAD3c1772tVx44YVZuXJl/u7v/i4f+tCHctJJJ+WRRx7p1v6jR4/Obrvtlne84x054IADsv322+cXv/hFkuTWW2/NUUcdlYEDB260z3HHHZcHHngga9eu7druf+/Pnj9/fs4888yMHDky2267bc4555z88Y9/zEMPPZT//M//zA477JC/+Zu/Sb9+/XLIIYfk4IMPLvhqAPR8Ahugh/vfK8rf+973smTJkvzkJz/JLrvsklNPPTX//d///ab7v+td7+r675qamkyePDn//u//nra2ttx+++05/vjjX7PPbrvtllGjRuWOO+7IE088kaeffjpHHHFEkuTZZ5/NpZdemrFjx2bs2LHZf//9s379+qxatSovvPBCdtppp42Oteuuu27iKwCwZRHYAD3Yvffem7Fjx2bdunVJ/ie2995771x++eVZt25d1q9fnyRpb2/v2qexsfHPHnPy5Mn5xS9+kbvvvjvbbbddxo4d+7rbHXfccbnjjjuycOHCja5y77TTTrnyyiuzZMmSrj8LFizIxIkTs9NOO+XZZ5/d6Dhr1qx5288fYEsksAF6sP333z+DBg3KxRdfnKeffjpJsnbt2lxzzTXZdddds88+++Qd73hHbrvttrS1teXuu+/O448//mePueuuu2bffffNFVdckcmTJ7/hdkceeWSefPLJjW4PSZLjjz8+3/72t/P000+ns7MzCxYsyMc//vE8//zzOfTQQ7Nu3bpcd911aW1tzf3335+77767zIsBsIUQ2AA9WG1tbX74wx+mtrY2J598cvbbb79MmDAhq1atyrx587Ltttvmkksuyb/9279l//33z6233ppjjjnmTY973HHHZfXq1Zk0adIbbjNw4MAcccQRGTBgwEZXuT/72c/m8MMPzymnnJLRo0dn3rx5ueaaa7L77rtn8ODBufbaa3PXXXdl//33z+zZs3PYYYcVeS0AthQ1nZ2dndUeAgAAegtXsAEAoCCBDQAABQlsAAAoSGADAEBBAhsAAAoS2AAAUJDABgCAggQ2AAAUJLABAKCg/w98kJ+fb4zOMgAAAABJRU5ErkJggg==
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Most-Passengers-who-did-not-survive-belonged-to-Class-3-i.e-the-lowest-class.-Most-people-who-survived-belonged-to-Class-1,-then-Class-3-and-then-Class-2.">Most Passengers who did not survive belonged to Class 3 i.e the lowest class. Most people who survived belonged to Class 1, then Class 3 and then Class 2.<a class="anchor-link" href="#Most-Passengers-who-did-not-survive-belonged-to-Class-3-i.e-the-lowest-class.-Most-people-who-survived-belonged-to-Class-1,-then-Class-3-and-then-Class-2.">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[24]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># What was the age distribution on the titanic</span>
<span class="n">titanic_df</span><span class="p">[</span><span class="s1">&#39;Age&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">hist</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[24]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x127d4e9b0&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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H9sVzc3NeTvXgaKv72zoiKMKPUL+ZYUeoAAOws+5tbW1Tz8386mvfx/LN/vqPDvrnN64rw4F8r4MGjQoWlpacrd37doVWZbF4MGDY+DAgbFr167cfU1NTTF48OAOf+zRo0fv71idUl9fn7dzHSj6+s62bm8q9Aj5V1ToAQrgIPyci4uL+/RzM5/6+vexfLOvzrOzzinkvurq6vZ5fL/f5m3kyJHR0NCQu71p06YYPnx4DBkyZK/7Pn3JBQAA9Fb7HcjnnnturF27NtatWxeNjY1x1113xfnnnx8REeedd16sWLEiNmzYENu2bYva2trcfQAA0Jvt9yUWf/VXfxW33357/OQnP4mPPvooJkyYENdcc01ERJx22mkxd+7cqKysjE8++SQuuuiimD17drcNDQAAPaVTgVxVVbXH7UmTJsWkSZP2+dgZM2bEjBkz9nswAAAoBD9qGgAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABLFXfnFK1asiPnz5+9xrLGxMf7hH/4hnn322Vi5cmUUF//5FMcff3w8+OCDXTkdAAD0uC4F8vTp02P69Om528uWLYvHHnsspk2bFr/61a/innvuifLy8i4PCQAA+dKlQE698847cdddd8WDDz4Y/fr1i40bN8aoUaO668MDAEBedNs1yDU1NVFRURFf/vKXY9OmTdG/f/+YO3dujBs3Li677LLYtGlTd50KAAB6TLe8grxly5Z44oknYvXq1RERsWPHjigrK4trr702Ro4cGUuWLImrrroqHnnkkdw1yZ+nvr6+O8b6Qs3NzXk714Gir+9s6IijCj1C/mWFHqAADsLPubW1tU8/N/Opr38fyzf76jw765zeuK9uCeSVK1fGxIkTY9iwYRERccYZZ8Tdd9+du7+qqiqWLVsWDQ0NccIJJ3zhxxs9enR3jPWF6uvr83auA0Vf39nW7U2FHiH/igo9QAEchJ9zcXFxn35u5lNf/z6Wb/bVeXbWOYXcV11d3T6Pd8slFk8//XRMmTIld/u5556L5cuX5263t7dHW1tblJSUdMfpAACgx3Q5kLMsi9deey1OP/30PY7feuutsX79+mhpaYlFixbFqFGj4phjjunq6QAAoEd1+RKL7du3x44dO+JLX/pS7ti4ceOiuro6KisrY9u2bVFWVhY1NTVdPRUAAPS4Lgfy4YcfHuvXr9/r+KxZs2LWrFld/fAAAJBXftQ0AAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJLrlR00DUFj9iooOuh+lXlLcPw471E9oBbqfQAY4ALS2t8flP1td6DHyaumNUwo9AnCAcokFAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACSKCz1Ab/LJzpZoaW0r9Bh5U1LcPw47tKTQYwAA9CoCOdHS2hbfXfDrQo+RN0tvnFLoEQAAeh2XWAAAQEIgAwBAQiADAEBCIAMAQKLLgXznnXfGKaecEqWlpVFaWhpnn312RESsWbMmJk+eHKWlpVFdXR07d+7s8rAAANDTuhzI69evj7//+7+Pl156KV566aV49tlnY8uWLfHjH/84brvttnj22WejtbU1Fi9e3B3zAgBAj+qWQD7ppJP2OLZ69eoYO3ZslJeXx+DBg6O6ujoeeuihrp4KAAB6XJcCubGxMd55552oqamJ8ePHR0VFRbzyyivR0NAQxx13XO5xI0eOjK1bt8a2bdu6PDAAAPSkLv2gkK1bt0Z5eXl873vfi9NPPz0efvjhuPLKK+OrX/1qHH744bnHDRgwIIqKiqK5ublDH7e+vr4rY3VYc3PzHucaOuKovJy3t2htbe30rj+9s77mYPs9joiIrNADFIDP+aCwP9/DIvr+97F8s6/Os7PO6Y376lIgH3300XHvvffmbs+YMSP+9V//Nerq6mL48OG547t27Yosy2Lw4MEd+rijR4/uylgdVl9fv8e5tm5vyst5e4vi4uJO7/rTO+trDrbf44iIKCr0AAXgcz4o7M/3sIi+/30s3+yr8+yscwq5r7q6un0e79IlFvX19bFs2bI9jrW0tMS3v/3taGhoyB3btGlTDB8+PIYMGdKV0wEAQI/rUiAPHDgwampqYu3atdHW1hb33ntv7N69OyZNmhRr166NdevWRWNjY9x1111x/vnnd9fMAADQY7p0icXIkSPjtttui5tuuik2b94cJ510UixevDj++q//Om6//fb4yU9+Eh999FFMmDAhrrnmmu6aGQAAekyXAjkiYurUqTF16tS9jk+aNCkmTZrU1Q8PAAB55UdNAwBAQiADAEBCIAMAQKLL1yDTd/UrKur0+wIPHXFUn34v4ewg/GEKAEDnCOSDWGt7e1z+s9WFHiOvfvWT/1voEQCAXk4gA9An7c/fgkX07b8JKynuH4cdWlLoMeCAJ5AB6JMOxr8FW3rjlEKPAAcF/0gPAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABJdCuTf/OY3cdFFF0VZWVl84xvfiOeffz4iIq677ro49dRTo7S0NEpLS+PSSy/tlmEBAKCnFe/vL/zjH/8Y11xzTSxatCgmTJgQjzzySPzt3/5tPPnkk7F+/fq45557ory8vDtnBQCAHrffryBv3rw5LrjggjjnnHOiX79+MX369IiI+MMf/hAbN26MUaNGdduQAACQL/sdyCeffHLcdNNNuduvvvpqNDc3R//+/aN///4xd+7cGDduXFx22WWxadOmbhkWAAB62n5fYpF67733Ys6cOTFnzpzYuXNnlJWVxbXXXhsjR46MJUuWxFVXXRWPPPJIFBd37HT19fXdMdYXam5u3uNcQ0cclZfz9hpZoQcoAJ/zwcHnfHA4CD/n1tbWvP0Z+b8+/WclX8zOOqc37qvL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DQ0Ne+xu5MiRsXXr1ti2bVshRuxVTj755Ljppptyt1999dVobm6ON998084+x6GHHhq/+93v4qyzzorly5fH1VdfvdfX2bHHHpu79OJg9/LLL8eLL74Y3/3ud3PHPC/3rbGxMd55552oqamJ8ePHR0VFRbzyyiv29TneeOONOP744+OGG26IsWPHxsUXXxyNjY3xhz/8wc46oKamJioqKuLLX/6yr7PPMHTo0JgxY0Z8//vfj1NOOSV+/vOfxy233NIr93XQBXJjY2MMHDhwj2MDBw6MpqamAk3UO40YMWKvY01NTXvsbsCAAVFUVBLNYPsAAAR7SURBVBTNzc35HK3Xe++992LOnDkxZ86cyLLMzr5AeXl5vPLKK3HFFVdEZWVl7Ny5MwYNGpS7f9CgQfYVES0tLXHjjTfGggULon///rnjnpf7tnXr1igvL4/vfe978fTTT8ell14aV1555V5/BtjX//enP/0pnnrqqTjttNPimWeeie9+97tx9dVX+xrrgC1btsQTTzwRl112WUR4Xn6W1tbWGDRoUPzLv/xLvPzyyzF37tyYM2dOr9zXQRfIgwYNipaWlj2ONTc3+yl/HfDp3e3atSuyLIvBgwcXcKre5Y033oiKioqYPn16XH755XbWASUlJVFSUhJXXHFFNDU1xeDBg2PXrl25+//32MHuzjvvjK997Wt7XTLga2zfjj766Lj33nvjb/7mb6KkpCRmzJgRw4cPj7q6Ovv6DCUlJXH00UdHRUVFlJSUxPTp0+NLX/pSPP/883b2BVauXBkTJ07M/a2r5+W+/frXv4633347JkyYECUlJfGd73wnDjnkkHjhhRd63b4OukAeOXJkNDQ05G5/8sknsX379jjmmGMKN1Qf8endbdq0KYYPHx5Dhgwp3FC9yAsvvBCzZ8+OysrKmDt3bkTY2edZs2ZNzJkzJ3e7vb09du/eHSUlJXvs7NN/9XawWr16ddx7771RXl4eF1xwQUT8+dX3ww8/3NfYPtTX18eyZcv2ONbS0hLf/va37eszHHvssXv9e5z29vb4zne+Y2df4Omnn44pU6bkbvvev29btmyJ1tbWPY4dcsgh8c1vfrPX7eugC+Rx48bF+++/H48++mi0tLRETU1NTJw40SvIHXDuuefG2rVrY926ddHY2Bh33XVXnH/++YUeq1f44IMP4uqrr47rr78+Zs2alTtuZ5/tK1/5Svz2t7+N3/3ud7F79+6444474sQTT4zvf//7sWLFitiwYUNs27Ytamtr7Sz+/A8+6+rq4oUXXoiVK1dGxJ//T9mll17qa2wfBg4cGDU1NbF27dpoa2uLe++9N3bv3h2TJk2yr89w1llnRcSf396zvb09Hn744di6daudfYEsy+K1116L008/PXfM9/59Gz9+fKxbty7++7//O9rb2+PBBx+MP/3pTzF16tTet69CvX1GIb388svZ17/+9eyMM87ILr/88uzDDz8s9Ei91h133JF7m7csy7Inn3wymzJlSlZWVpbNmTMna2xsLOB0vceSJUuyE088MTvjjDP2+O/FF1+0s8/x9NNPZ+edd15WXl6eVVZWZlu2bMmyLMseeOCB7Ktf/WpWXl6e3XTTTVlra2uBJ+1d3n///dzbvGWZ5+VnWbVqVTZ16tTs9NNPzyoqKrI33ngjyzL7+jxvvvlm9q1vfSsrKyvLLrzwwqyuri7LMjv7PB9//HF24oknZrt27drjuJ3t26OPPpqdd955WVlZWVZRUZG9/vrrWZb1vn0VZVmWFTbRAQCg9zjoLrEAAIDPI5ABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAIPH/APVduJLjh2xvAAAAAElFTkSuQmCC
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="As-we-can-see,-the-titanic-was-populated-by-more-younger-people-of-age-&lt;-30.-Which-means-lots-of-children-and-young-adults.">As we can see, the titanic was populated by more younger people of age &lt; 30. Which means lots of children and young adults.<a class="anchor-link" href="#As-we-can-see,-the-titanic-was-populated-by-more-younger-people-of-age-&lt;-30.-Which-means-lots-of-children-and-young-adults.">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="What-were-the-ages-of-the-survivors?-What-was-the-age-distribution-on-the-titanic">What were the ages of the survivors? What was the age distribution on the titanic<a class="anchor-link" href="#What-were-the-ages-of-the-survivors?-What-was-the-age-distribution-on-the-titanic">&#182;</a></h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[25]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="p">[</span><span class="n">titanic_df</span><span class="p">[</span><span class="s1">&#39;Survived&#39;</span><span class="p">]</span> <span class="o">==</span> <span class="mi">1</span><span class="p">][</span><span class="s1">&#39;Age&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">hist</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[25]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x127ef6b70&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Ages-of-20-40-were-more-likely-to-survive,-followed-by-Ages-below-5.-We-would-be-inclined-to-think-that-children-were-more-likelt-to-survive-but-Ages-10-20-have-a-lower-number.-We-need-to-ask-ourself-if-that-is-because-there-were-a-low-number-of-population-for-ages-10-20.">Ages of 20-40 were more likely to survive, followed by Ages below 5. We would be inclined to think that children were more likelt to survive but Ages 10-20 have a lower number. We need to ask ourself if that is because there were a low number of population for ages 10-20.<a class="anchor-link" href="#Ages-of-20-40-were-more-likely-to-survive,-followed-by-Ages-below-5.-We-would-be-inclined-to-think-that-children-were-more-likelt-to-survive-but-Ages-10-20-have-a-lower-number.-We-need-to-ask-ourself-if-that-is-because-there-were-a-low-number-of-population-for-ages-10-20.">&#182;</a></h3><h4 id="(There-were-close-to-40-passengers-from-ages-10-20-and-close-to-20-survived)">(There were close to 40 passengers from ages 10-20 and close to 20 survived)<a class="anchor-link" href="#(There-were-close-to-40-passengers-from-ages-10-20-and-close-to-20-survived)">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Analysing-the-fare-distribution.-How-many-people-paid-what-sums-of-fare-on-the-ship?">Analysing the fare distribution. How many people paid what sums of fare on the ship?<a class="anchor-link" href="#Analysing-the-fare-distribution.-How-many-people-paid-what-sums-of-fare-on-the-ship?">&#182;</a></h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[26]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="p">[</span><span class="s1">&#39;Fare&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">bins</span><span class="o">=</span><span class="mi">20</span><span class="p">,</span><span class="n">color</span><span class="o">=</span><span class="s1">&#39;y&#39;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[26]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x127faa320&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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yCFxyCGHxPz58+PMM8+Mq6++OjZu3BgREW1tbTFmzJj8x4wePTpaW1sHYfsAADCwSr7Ii999992YN29ezJs3L7Zv3x4TJkyIW265JSorK+Oxxx6LuXPnxrPPPhudnZ1RXl6e/7jy8vLIZrP9XieTyXyRbQ2YUccdUZB1e3t7o7VA93wwyGazBfuaYnCZbfEy2+JltsWrmGbb70B+66234pprrolLLrkkZs+eHRERjz/+eP76DTfcEE888US0tbVFWVlZdHV15a91dnZGRUVFvzc1fvz4fr92IHVs21qQdUtKSgp2zweDTCbj81ukzLZ4mW3xMtvidSDOtqWlZbfn+/UuFmvXro26urqor6+P+fPnR0TEb37zm1ixYkX+NTt27Ii+vr4oLS2NysrKaGtry1/79CMXAACwv9pjIL///vtx3XXXxe233x5XXnnlTtfuu+++WL9+fXR3d8dDDz0UJ554YowaNSqmT58eTz/9dGzYsCHa29ujqakpZsyYMWg3AQAAA2WPgbxixYr4+OOP45577sm/33FVVVUMGzYsGhoaor6+PiZNmhTr16+PxsbGiIg47bTTYv78+VFfXx/f/OY346STToq6urpBvxkAANhbe3wG+dprr41rr712t9eqqqp2+any/6qtrY3a2tq92x0AAOxjftU0AAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJPoVyC+99FLMmjUrJkyYEBdffHGsWbMmIiKam5tj2rRpUVVVFQ0NDbF9+/b8xyxdujRqampi4sSJsWDBgujr6xucOwAAgAG0x0D+6KOP4qabboqbb7451q5dG7Nnz47rr78+3nvvvbj11lvj/vvvj1deeSV6e3vj0UcfjYiIN998MxobG2Px4sXR3NwcmUwmli1bNug3AwAAe2uPgbxly5a48MIL45xzzomhQ4fGRRddFBERzz33XEyaNCmqq6ujoqIiGhoaYvny5RERsXLlypg5c2aMHTs2RowYEXPmzMlfAwCA/dkeA/mkk06Ku+++O3+8bt26yGaz8fbbb8eYMWPy5ysrK2Pr1q3R3t4ebW1tO10bPXp0tLa2DvDWAQBg4JV8kRe/++67MW/evJg3b160trZGWVlZ/tqwYcNiyJAhkc1mo7OzM8rLy/PXysvLI5vN9nudTCbzRbY1YEYdd0RB1u3t7Y3WAt3zwSCbzRbsa4rBZbbFy2yLl9kWr2Kabb8D+a233oprrrkmLrnkkpg9e3bce++90d3dnb/e1dUVuVwuKioqoqysLLq6uvLXOjs7o6Kiot+bGj9+fL9fO5A6tm0tyLolJSUFu+eDQSaT8fktUmZbvMy2eJlt8ToQZ9vS0rLb8/16F4u1a9dGXV1d1NfXx/z58yPiD49UtLW15V+zcePGGDlyZAwfPnyXa59+5AIAAPZXewzk999/P6677rq4/fbb48orr8yfP++882L16tXx6quvRkdHRyxatChmzJgRERHTp0+Pp59+OjZs2BDt7e3R1NSUvwYAAPuzPQbyihUr4uOPP4577rknqqqq8v/bvHlzPPjgg3HnnXdGTU1NlJSUxE033RQREaeddlrMnz8/6uvr45vf/GacdNJJUVdXN+g3AwAAe2uPzyBfe+21ce21137m9alTp+72fG1tbdTW1n7pjQEAQCH4VdMAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQ+EKB3NTUFHfccUf++LbbbotTTz01qqqqoqqqKi699NL8taVLl0ZNTU1MnDgxFixYEH19fQO3awAAGCQl/XlRd3d3/PjHP47HHnssLrnkkvz59evXx09/+tOorq7e6fVvvvlmNDY2xs9+9rP46le/GnPmzIlly5bFFVdcMbC7LxZDhkR39uN9vuzQoV+JktKKfb4uAMD+rF+BvGDBgvjwww/j8ssvj56enoiI6Ovri9bW1jjxxBN3ef3KlStj5syZMXbs2IiImDNnTvzDP/yDQP4sO3bEul//cJ8ve+qU7+/zNQEA9nf9esTixhtvjKampjj88MPz5zZu3BiHHHJIzJ8/P84888y4+uqrY+PGjRER0dbWFmPGjMm/dvTo0dHa2jrAWwcAgIHXr58gH3nkkbuc27ZtW0yYMCFuueWWqKysjMceeyzmzp0bzz77bHR2dkZ5eXn+teXl5ZHNZvu9qUwm0+/XDqRRxx1RkHVzkSvIur29vdFaoM/1vpTNZgv2NcXgMtviZbbFy2yLVzHNtl+BvDtnnHFGPP744/njG264IZ544oloa2uLsrKy6Orqyl/r7OyMior+P+s6fvz4L7utvdKxbWtB1h0SQwqybklJScE+1/tSJpM5KO7zYGS2xctsi5fZFq8DcbYtLS27Pf+l3+btN7/5TaxYsSJ/vGPHjujr64vS0tKorKyMtra2/LVPP3IBAAD7q716H+T77rsv1q9fH93d3fHQQw/FiSeeGKNGjYrp06fH008/HRs2bIj29vZoamqKGTNmDNSeAQBg0HzpRyzOPPPMaGhoiPr6+mhvb48JEyZEY2NjREScdtppMX/+/Kivr49PPvkkZs2aFXV1dQO2aQAAGCxfKJBvuOGGnY6vvPLKuPLKK3f72tra2qitrf3yOwMAgALwq6YBACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAg8YUCuampKe644478cXNzc0ybNi2qqqqioaEhtm/fnr+2dOnSqKmpiYkTJ8aCBQuir69v4HYNAACDpF+B3N3dHQ8//HA89NBD+XPvvfde3HrrrXH//ffHK6+8Er29vfHoo49GRMSbb74ZjY2NsXjx4mhubo5MJhPLli0bnDsAAIAB1K9AXrBgQfzud7+Lyy+/PH/u+eefj0mTJkV1dXVUVFREQ0NDLF++PCIiVq5cGTNnzoyxY8fGiBEjYs6cOflrAACwP+tXIN94443R1NQUhx9+eP5cW1tbjBkzJn9cWVkZW7dujfb29l2ujR49OlpbWwdw2wAAMDhK+vOiI488cpdznZ2dcdhhh+WPhw0bFkOGDIlsNhudnZ1RXl6ev1ZeXh7ZbLbfm8pkMv1+7UAaddwRBVk3F7mCrNvb2xutBfpc70vZbLZgX1MMLrMtXmZbvMy2eBXTbPsVyLtTXl4e3d3d+eOurq7I5XJRUVERZWVl0dXVlb/W2dkZFRUV/f5njx8//stua690bNtakHWHxJCCrFtSUlKwz/W+lMlkDor7PBiZbfEy2+JltsXrQJxtS0vLbs9/6bd5q6ysjLa2tvzxxo0bY+TIkTF8+PBdrn36kQsAANhffelAPu+882L16tXx6quvRkdHRyxatChmzJgRERHTp0+Pp59+OjZs2BDt7e3R1NSUvwYAAPuzL/2IxdFHHx0PPvhg3HnnnfHhhx9GTU1N3HTTTRERcdppp8X8+fOjvr4+Pvnkk5g1a1bU1dUN2KYBAGCwfKFAvuGGG3Y6njp1akydOnW3r62trY3a2tovvTEAACgEv2oaAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEnsdyI888kiccsopUVVVFVVVVXH22WdHRERzc3NMmzYtqqqqoqGhIbZv377XmwUAgMFWsrf/gPXr18e9994bF198cf7ce++9F7feems89thjcdJJJ8XNN98cjz76aNx88817uxwDaciQ6M5+XJClhw79SpSUVhRkbQCAzzMggXz99dfvdO7555+PSZMmRXV1dURENDQ0xOzZswXy/mbHjlj36x8WZOlTp3y/IOsCAOzJXj1i0dHREe+88040NjbG5MmT47LLLos33ngj2traYsyYMfnXVVZWxtatW6O9vX2vNwwAAINpr36CvHXr1qiuro7vfve7cfrpp8fPf/7zmDNnTnzjG9+Iww47LP+6YcOGxZAhQyKbzfbrn5vJZPZmW1/aqOOOKMi6ucgdVOtGRPT29kbrPppzNpst2NcUg8tsi5fZFi+zLV7FNNu9CuSvfe1rsWTJkvxxbW1t/NM//VO0tLTEyJEj8+e7uroil8tFRUX/njkdP3783mzrS+vYtrUg6w6JIQfVuhERJSUl+2zOmUymYF9TDC6zLV5mW7zMtngdiLNtaWnZ7fm9esQik8nEE088sdO57u7u+M53vhNtbW35cxs3boyRI0fG8OHD92Y5AAAYdHsVyGVlZdHY2BirV6+Ovr6+WLJkSfT09MTUqVNj9erV8eqrr0ZHR0csWrQoZsyYMVB7BgCAQbNXj1hUVlbG/fffH3fffXds2bIlxo0bF48++mgce+yx8eCDD8add94ZH374YdTU1MRNN900UHsGAIBBs9dv83b++efH+eefv8v5qVOnxtSpU/f2Hw8AAPuUXzUNAAAJgQwAAAmBDAAACYEMAAAJgQwAAAmBDAAACYEMAAAJgQwAAAmBDAAAib3+TXrwpQwZEt3Zj/fJUv/va0fm1xo69CtRUlqxT9YFAA5MApnC2LEj1v36h/t82VOnfH+frwkAHFg8YgEAAAmBDAAACYEMAAAJgQwAAAmBDAAACYEMAAAJgQwAAAmBDAAACYEMAAAJgQwAAAmBDAAACYEMAAAJgQwAAAmBDAAACYEMAAAJgQwAAAmBDAAACYEMAACJkkJvAPapIUOiO/txgZYeGrncjn2+7tChX4mS0op9vi4AHKgEMgeXHTti3a9/WJClT625oyBrnzrl+/t8TQA4kHnEAgAAEgIZAAASAhkAABICGQAAEgIZAAAS3sUCKDq93R2xY0fPPl/XW+oBFAeBDBSdHTt6Yt1L9+7zdQv5lnr+pQBg4AhkgCJwMP5LAcBgGbRnkFtaWmLWrFlxxhlnxNVXXx3vv//+YC0FAAADZlB+gpzNZqOhoSHuvPPOOPfcc+O+++6L++67Lx5++OHBWA4AgL20t49q/b+vHRnd2Y+/8Mftj49qDUogr169OkaOHBkXXHBBRETceOONcfbZZ8e2bdvij/7ojwZjSeCzDBnypb5hDYT98ZveoCrg5zoiV6B1gWLhUa3/MyiBvGnTphgzZkz+ePjw4TF8+PDYtGlTnHzyyYOxJPBZduyIdb/+YUGW3h+/6Q2qQn6ua+4oyLoAxWhILpcb8B87/OQnP4n/+Z//ifvuuy9/7rzzzosHHnggqqurP/djW1paBno7AACwWxMnTtzl3KD8BLm8vDy6u7t3OpfNZuPQQw/d48fubpMAALCvDMq7WFRWVkZbW1v++JNPPomPP/44Ro0aNRjLAQDAgBmUQD7zzDNj8+bN8Ytf/CK6u7ujsbExpkyZ0q+fIAMAQCENyjPIERH/+Z//GQsWLIhNmzbFhAkT4oEHHoiRI0cOxlIAADBgBi2QAQDgQDRov0kPAAAORAIZAAASAjn+8N7Ls2bNijPOOCOuvvrqeP/99wu9Jb6gpqamuOOO//tFCc3NzTFt2rSoqqqKhoaG2L59e/7a0qVLo+3gExgAAAZ0SURBVKamJiZOnBgLFiyIvr6+QmyZPXjppZdi1qxZMWHChLj44otjzZo1EWG2xeCpp56Kc889N6qqqqKuri5aW1sj4vPnt2jRopg8eXJMmjQpFi5cWKit009r1qyJcePG5Y/9uT3wPfLII3HKKadEVVVVVFVVxdlnnx0RRTzb3EGus7Mzd9ZZZ+X+7d/+LdfV1ZW76667cvPnzy/0tuinrq6u3N/93d/lTjzxxNztt9+ey+VyuS1btuQmTJiQW7NmTW779u25uXPn5h588MFcLpfLrVu3LnfmmWfm/uu//iv30Ucf5Wpra3P//M//XMhbYDe2bt2amzhxYu4//uM/cn19fbmf//znua9//etmWwQymUzu61//eu7tt9/O9fX15RobG3N/+Zd/+bnz+/d///fct771rdyWLVty77zzTm7atGm5F198scB3wmfp7OzMnX/++bkTTjghl8v5nlws/uqv/ir3r//6rzudK+bZHvQ/QV69enWMHDkyLrjggigtLY0bb7wxmpubY9u2bYXeGv2wYMGC+N3vfheXX355/tzzzz8fkyZNiurq6qioqIiGhoZYvnx5RESsXLkyZs6cGWPHjo0RI0bEnDlz8tfYf2zZsiUuvPDCOOecc2Lo0KFx0UUXRUTEc889Z7YHuHHjxsULL7wQxx9/fLS3t8e2bdtixIgRnzu/Z599Nq644oo46qij4thjj426ujqz3Y81NjZGTU1N/tj35OKwfv36nf5WIKK4Z3vQB/KmTZtizJgx+ePhw4fH8OHDY9OmTQXcFf114403RlNTUxx++OH5c21tbTvNtLKyMrZu3Rrt7e27XBs9enT+r3fZf5x00klx991354/XrVsX2Ww23n77bbMtAoceemisWrUqzjrrrFixYkVcd911nzs/sz1wvP766/Haa6/FVVddlT/ne/KBr6OjI955551obGyMyZMnx2WXXRZvvPFGUc/2oA/kjo6OKCsr2+lcWVlZdHZ2FmhHfBFHHnnkLuc6Ozt3mumwYcNiyJAhkc1mo7OzM8rLy/PXysvLI5vN7pO98uW8++67MW/evJg3b17kcjmzLRLV1dXxxhtvxDXXXBP19fWxffv2z5yf2R4Yuru74wc/+EHcc889ccghh+TP+5584Nu6dWtUV1fHd7/73XjxxRfj0ksvjTlz5uzSUMU024M+kMvLy6O7u3unc9ls1m/9O4B9eqZdXV2Ry+WioqIiysrKoqurK3+ts7MzKioqCrFN+uGtt96Kyy67LC666KKYPXu22RaR0tLSKC0tjWuuuSY/q8+an9keGB555JE499xzd/lreH9uD3xf+9rXYsmSJfGnf/qnUVpaGrW1tTFy5MhoaWkp2tke9IFcWVkZbW1t+eNPPvkkPv744xg1alThNsVe+fRMN27cGCNHjozhw4fvcu3TfwXE/mPt2rVRV1cX9fX1MX/+/Igw22LQ3Nwc8+bNyx/v2LEjenp6orS09DPnV1lZudNjb2a7f3r++edjyZIlUV1dHRdeeGFE/OFvCg477DB/bg9wmUwmnnjiiZ3OdXd3x3e+852ine1BH8hnnnlmbN68OX7xi19Ed3d3NDY2xpQpU/wE+QB23nnnxerVq+PVV1+Njo6OWLRoUcyYMSMiIqZPnx5PP/10bNiwIdrb26OpqSl/jf3H+++/H9ddd13cfvvtceWVV+bPm+2B7+STT46XX345Vq1aFT09PbFw4cI44YQT4nvf+95nzm/GjBmxZMmS2Lx5c7z77rvx5JNP5gOM/cdzzz0XLS0tsXbt2li5cmVE/OFfdC+99FJ/bg9wZWVl0djYGKtXr46+vr5YsmRJ9PT0xNSpU4t3tgV9D439xOuvv5778z//89wZZ5yRmz17du6DDz4o9Jb4ghYuXJh/m7dcLpf71a9+lfvWt76VmzBhQm7evHm5jo6O/LVly5blvvGNb+Sqq6tzd999d663t7cQW+ZzPPbYY7kTTjghd8YZZ+z0v9dee81si8CLL76Ymz59eq66ujpXX1+fe++993K53OfP7yc/+Unuz/7sz3KTJk3KLVq0qFBbp582b96cf5u3XM735GLw3HPP5c4///zc6aefnrvssstyb731Vi6XK97ZDsnlcrlCRzoAAOwvDvpHLAAAICWQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACDx/wFS6QTUK7LL8wAAAABJRU5ErkJggg==
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="What-were-the-Fares-of-the-survivors?">What were the Fares of the survivors?<a class="anchor-link" href="#What-were-the-Fares-of-the-survivors?">&#182;</a></h2><h3 id="Fare-distribution-on-the-titanic-below--">Fare distribution on the titanic below -<a class="anchor-link" href="#Fare-distribution-on-the-titanic-below--">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[27]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="p">[</span><span class="n">titanic_df</span><span class="p">[</span><span class="s1">&#39;Survived&#39;</span><span class="p">]</span> <span class="o">==</span> <span class="mi">1</span><span class="p">][</span><span class="s1">&#39;Fare&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">color</span><span class="o">=</span><span class="s1">&#39;y&#39;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[27]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x128185358&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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+s1R32IxZcqUaGhoiLq6uti9e3dUVVVFU1NTREScffbZsXjx4qirq4sPPvggZs+eHbW1tX22aQAA6C+fKJDr6+sPOV6wYEEsWLDgiI+dO3duzJ079+h3BgAABeCtpgEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAg8YkCubm5OW655Zb88dq1a+PCCy+MSZMmxV//9V/H1q1b8+dqa2vjnHPOicrKyqisrIyGhoa+2zUAAPSTkp48qKurK374wx/GqlWr4vLLL4+IiFdffTWWL18eP/nJT2LixIlx3333RX19fTz22GMREbF169bYuHFjnHLKKf23ewAA6GM9ega5sbExXn755Zg/f35+bceOHVFbWxtnnXVWDBs2LBYsWBBtbW3xwQcfxM6dOyOXy4ljAAAGnR4F8pIlS6K5uTlGjRqVX5s2bdoht0388pe/jFNOOSX++I//OFpbW2P48OExf/78mDp1atTX18e7777b97sHAIA+1qNbLMaMGfOR5zOZTDQ2NsbSpUsjImL//v1x9tlnx0033RQnnXRS3HbbbXHjjTfGAw880KNNZTKZHj2ur4397MkFuW6hHDhwILYV6Hs9kLLZbMF+puhfZlu8zLZ4mW3xKqbZ9iiQP8rmzZtj0aJF8Q//8A8xc+bMiIiYMWNGzJgxI/+YG264IaZOnRrt7e1RXl7+sZ9zwoQJvd3WUWnfu6sg1y2UkpKSgn2vB1Imkzkuvs7jkdkWL7MtXmZbvAbjbFtaWo643quXedu4cWPU19fH8uXLY968efn1xx57LJ544on88f79+2Po0KFRUtLrHgcAgH511MX6yiuvxE033RT33ntvTJ069ZBze/fujXvvvTfOOOOMOPHEE+POO++MmTNnRmlpaa83DAAA/emoA/mnP/1pdHZ2xnXXXXfI+saNG+PrX/96vPnmmzFv3rxob2+PadOmxbJly3q9WQAA6G+fKJDr6+vz/75s2bKPjN6GhgZvDgIAwKDjraYBACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAICGQAQAgIZABACAhkAEAIPGJArm5uTluueWW/PGmTZtixowZUVlZGQ0NDbFv3778uQcffDBqampi0qRJ0djYGN3d3X23awAA6Cc9CuSurq74wQ9+ECtWrMivvf322/Gd73wn7rjjjnj66afjwIEDce+990ZExEsvvRRNTU3xwAMPxKZNmyKTycRDDz3UP18BAAD0oR4FcmNjY7z88ssxf/78/Nrjjz8ekydPjurq6igvL4+GhoZ45JFHIiJiw4YNcemll8Zpp50WJ554YixcuDB/DgAAjmU9CuQlS5ZEc3NzjBo1Kr/W1tYW48aNyx9XVFTErl27Yvfu3YedO/XUU2Pbtm19uG0AAOgfJT150JgxYw5b6+joiBNOOCF/PHz48BgyZEhks9no6OiIsrKy/LmysrLIZrM93lQmk+nxY/vS2M+eXJDrFsqBAwdiW4G+1wMpm80W7GeK/mW2xctsi5fZFq9imm2PAvlIysrKoqurK3/c2dkZuVwuysvLY8SIEdHZ2Zk/19HREeXl5T3+3BMmTDjabfVK+95dBbluoZSUlBTsez2QMpnMcfF1Ho/MtniZbfEy2+I1GGfb0tJyxPWjfpm3ioqKaGtryx9v3749Ro8eHSNHjjzs3B/ecgEAAMeqow7kCy64IDZv3hzPPvtstLe3x8qVK2PWrFkREXHxxRfHunXrYuvWrbF79+5obm7OnwMAgGPZUd9i8Wd/9mdx1113xa233hrvvfde1NTUxPXXXx8REWeffXYsXrw46urq4oMPPojZs2dHbW1tn20aAAD6yycK5Pr6+kOOp0+fHtOnTz/iY+fOnRtz58496o0BAEAheKtpAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEiU9OaD161bF42NjYestbe3xz//8z/H008/HRs2bIiSkt9f4rTTTouHH364N5cDAIB+16tAnjNnTsyZMyd/vHr16nj00Udj5syZ8ZOf/CTuv//+qK6u7vUmAQBgoPQqkFNvvPFGrFy5Mh5++OEYOnRobNu2LcaPH99Xnx4AAAZEn92D3NTUFPPmzYvPf/7zsX379hg2bFgsXrw4pkyZEldeeWVs3769ry4FAAD9pk+eQX777bfjF7/4RTz++OMREbF3796oqqqKG2+8MSoqKmLVqlVx7bXXxvr16/P3JH+UTCbTF9v6xMZ+9uSCXLdQDhw4ENsK9L0eSNlstmA/U/Qvsy1eZlu8zLZ4FdNs+ySQN2zYENOmTYtRo0ZFRMS5554b9913X/58fX19rF69Otra2uL000//2M83YcKEvtjWJ9a+d1dBrlsoJSUlBfteD6RMJnNcfJ3HI7MtXmZbvMy2eA3G2ba0tBxxvU9usXjyySfjwgsvzB8/88wzsXbt2vzxwYMHo7u7O0pLS/vicgAA0G96Hci5XC5eeumlOOeccw5Zv/3226O1tTW6urpixYoVMX78+Bg7dmxvLwcAAP2q17dY7NmzJ/bu3Rsnn/z/79+dMmVKNDQ0RF1dXezevTuqqqqiqampt5cCAIB+1+tAPuGEE6K1tfWw9QULFsSCBQt6++kBAGBAeatpAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEiUFHoDFNCQIdGV3VPoXfS7z39uTP7rHDr0U1FSWl7gHQEAxzKBfDw7eDC2/HJ5oXcxoM6a9t1CbwEAOMa5xQIAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABICGQAAEgIZAAASAhkAABK9DuR77rknvvjFL0ZlZWVUVlbGeeedFxERmzZtihkzZkRlZWU0NDTEvn37er1ZAADob70O5NbW1vj+978fL7zwQrzwwgvx9NNPx9tvvx3f+c534o477oinn346Dhw4EPfee29f7BcAAPpVnwTymWeeecja448/HpMnT47q6uooLy+PhoaGeOSRR3p7KQAA6He9CuT29vZ44403oqmpKaZOnRrz5s2LF198Mdra2mLcuHH5x1VUVMSuXbti9+7dvd4wAAD0p5LefPCuXbuiuro6vvWtb8U555wTP/vZz2LhwoXxla98JU444YT844YPHx5DhgyJbDbbo8+byWR6s62jNvazJxfkuoWSi1yhtzDgDhw4ENsK9PNF38tmswX7+4L+ZbbFy2yLVzHNtleB/LnPfS7WrFmTP547d27867/+a7S0tMTo0aPz652dnZHL5aK8vLxHn3fChAm92dZRa9+7qyDXLZQhMaTQWxhwJSUlBfv5ou9lMhnzLFJmW7zMtngNxtm2tLQccb1Xt1hkMplYvXr1IWtdXV3xzW9+M9ra2vJr27dvj9GjR8fIkSN7czkAAOh3vQrkESNGRFNTU2zevDm6u7tjzZo1sX///pg+fXps3rw5nn322Whvb4+VK1fGrFmz+mrPAADQb3p1i0VFRUXccccdsXTp0ti5c2eceeaZce+998ZnPvOZuOuuu+LWW2+N9957L2pqauL666/vqz0DAEC/6VUgR0RcdNFFcdFFFx22Pn369Jg+fXpvPz0AAAwobzUNAAAJgQwAAAmBDAAACYEMAAAJgQwAAAmBDAAACYEMAAAJgQwAAAmBDAAACYEMAAAJgQwAAAmBDAAACYEMAAAJgQwAAAmBDAAACYEMAAAJgQwAAAmBDAAACYEMAAAJgQwAAAmBDAAACYEMAAAJgQwAAAmBDAAACYEMAAAJgQwAAAmBDAAACYEMAAAJgQwAAAmBDAAACYEMAAAJgQwAAAmBDAAACYEMAAAJgQwAAAmBDAAACYEMAAAJgQwAAAmBDAAACYEMAAAJgQwAAAmBDAAACYEMAAAJgQwAAIleBfJTTz0Vs2fP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woUL8+c4duzcuTMuueSSOP/882Po0KExZ86ciIjYuHGj2Q5yZ555ZvziF7+I008/PXbv3h179+6NE0888SPnt379+vjGN74Rf/qnfxqf+cxnora21myPYU1NTVFTU5M/9ndycWhtbT3ktwIRxT3b4z6QX3vttRg3blz+eOTIkTFy5Mh47bXXCrgremrJkiXR3Nwco0aNyq+1tbUdMtOKiorYtWtX7N69+7Bzp556av7Xuxw7Jk6cGEuXLs0fb9myJbLZbLz66qtmWwQ+/elPx69//ev48pe/HGvXro3rrrvuI+dntoPHb3/72/jNb4NdySYAAAPHSURBVH4Tf/u3f5tf83fy4Nfe3h5vvPFGNDU1xdSpU2PevHnx4osvFvVsj/tAbm9vjxEjRhyyNmLEiOjo6CjQjvgkxowZc9haR0fHITMdPnx4DBkyJLLZbHR0dERZWVn+XFlZWWSz2QHZK0fnrbfeikWLFsWiRYsil8uZbZGorq6OF198Ma6++uqoq6uLffv2fej8zHZw6Orqin/8x3+MZcuWxbBhw/Lr/k4e/Hbt2hXV1dXxrW99K5588sm44oorYuHChYc1VDHN9rgP5LKysujq6jpkLZvNete/QewPZ9rZ2Rm5XC7Ky8tjxIgR0dnZmT/X0dER5eXlhdgmPfDKK6/EvHnzYs6cOXHVVVeZbREpLS2N0tLSuPrqq/Oz+rD5me3gcM8998RXv/rVw34N78/t4Pe5z30u1qxZE3/+538epaWlMXfu3Bg9enS0tLQU7WyP+0CuqKiItra2/PEHH3wQe/bsibFjxxZuU/TKH850+/btMXr06Bg5cuRh5/7wV0AcO55//vmora2Nurq6WLx4cUSYbTHYtGlTLFq0KH988ODB2L9/f5SWln7o/CoqKg657c1sj02PP/54rFmzJqqrq+OSSy6JiN//puCEE07w53aQy2QysXr16kPWurq64pvf/GbRzva4D+QpU6bEjh074uc//3l0dXVFU1NTTJs2zTPIg9gFF1wQmzdvjmeffTba29tj5cqVMWvWrIiIuPjii2PdunWxdevW2L17dzQ3N+fPcex455134rrrroubb745FixYkF8328HvC1/4QvzqV7+KX//617F///64++6744wzzoi/+7u/+9D5zZo1K9asWRM7duyIt956K/793/89H2AcOzZu3BgtLS3x/PPPx4YNGyLi9/+he8UVV/hzO8iNGDEimpqaYvPmzdHd3R1r1qyJ/fv3x/Tp04t3tgV9DY1jxG9/+9vc1772tdy5556bu+qqq3LvvvtuobfEJ3T33XfnX+Ytl8vlnnjiidyFF16Yq6qqyi1atCjX3t6eP/fQQw/lvvKVr+Sqq6tzS5cuzR04cKAQW+YjrFq1KnfGGWfkzj333EP++c1vfmO2ReDJJ5/MXXzxxbnq6upcXV1d7u23387lch89vx/96Ee5v/iLv8hNnjw5t3LlykJtnR7asWNH/mXecjl/JxeDjRs35i666KLcOeeck5s3b17ulVdeyeVyxTvbIblcLlfoSAcAgGPFcX+LBQAApAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACQEMgAAJAQyAAAkBDIAACT+Hzsf1qOzCg9+AAAAAElFTkSuQmCC
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Analysing-how-many-survived-from-the-three-embarking-stations">Analysing how many survived from the three embarking stations<a class="anchor-link" href="#Analysing-how-many-survived-from-the-three-embarking-stations">&#182;</a></h1><h3 id="S---Southamption">S - Southamption<a class="anchor-link" href="#S---Southamption">&#182;</a></h3><h3 id="C---Cherbourg">C - Cherbourg<a class="anchor-link" href="#C---Cherbourg">&#182;</a></h3><h3 id="Q---Queenstown">Q - Queenstown<a class="anchor-link" href="#Q---Queenstown">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[28]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sns</span><span class="o">.</span><span class="n">countplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="s2">&quot;Survived&quot;</span><span class="p">,</span> <span class="n">hue</span><span class="o">=</span><span class="s2">&quot;Embarked&quot;</span><span class="p">,</span> <span class="n">data</span><span class="o">=</span><span class="n">titanic_df</span><span class="p">,</span> <span class="n">palette</span><span class="o">=</span><span class="s2">&quot;Set1&quot;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[28]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x12834fe80&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAtgAAAHlCAYAAADP34vrAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjAsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+17YcXAAAgAElEQVR4nO3de5SXdb33/xcwATMWQSJWKgHeqWhbOZmaGzQPWwFJ0mzr7eZeHgNREM2zd2Ee0pax5BY8JFke77209qSIh51Yu4OyNzm3h1yitLkZUlBAaYjDwMww8/uj3547EmuQ6zvfYXw81nItv9f1va7rPTOrWU+vPnN9u7S0tLQEAAAoRNdyDwAAAJ2JwAYAgAIJbAAAKJDABgCAAglsAAAokMAGAIACVZR7gKLV1NSUewQAAD4khg8f/p5tnS6wk21/oQAAUKT3u7FriQgAABRIYAMAQIEENgAAFEhgAwBAgTrlHzkCAFC85ubmrFixIo2NjeUepd307t07vXv3TpcuXdp8jMAGAKBNVqxYkV69eqVXr17lHqVdNDc3Z9WqVXn77bfzqU99qs3HWSICAECbNDY2fmjiOkm6du2a3XffPfX19dt3XInmAQCAnV6XLl22a3lIIrABAKBQAhsAgB3y5ptv5oADDsiJJ5641T/f/OY323T8f/zHf+SrX/3qDs1wxRVX5J//+Z8/8PFHHXVUlixZskMz/Bd/5AgAwA7r3bt3Hn300XKP0SEIbAAASmbChAk54IADUlNTk3feeSdXX3115s2bl1dffTX77bdfZs6cmSRZu3ZtJk2alOXLl2fQoEG5/vrr87GPfSxPP/107r777mzevDnr1q3LxRdfnDFjxmTWrFl58cUXs3Llyhx99NGt12tqasqFF16YT3/607n66qvz2muv5YYbbsj69evTo0ePXHnllTnooIPy1ltv5dJLL83atWuzzz77ZPPmzYV9zQIbAIAdVldXlxNPPHGrbRMnTkzyp3j+0Y9+lB//+Me55JJL8uSTT6Zv374ZM2ZMXnnllSTJ8uXLM3v27Hz2s5/Nddddl9tvvz2XXXZZHnjggdx6663p169fFixYkOnTp2fMmDGt13zsscfSpUuXXHHFFWlubs7Xv/719O/fP5dffnkaGxtz2WWXZfbs2enfv3+WLFmSc845J08//XSuu+66HHPMMTnjjDOycOHCzJs3r7DvhcAGAGCHvd8SkX/+539uvcO81157ZdCgQa3PlP7kJz+Zurq69OjRIwceeGA++9nPJklOPPHETJ8+PV26dMltt92Wn//851m2bFlefvnlbNy4sfXcQ4YM2eoJH7fffns2btyYZ555JklSW1ubZcuWZcqUKa3vaWpqyltvvZUFCxbkuuuuS5J8/vOfz1577VXY90JgAwBQUt27d2/994qKbednt27dWv+9ubk53bp1y4YNG3LyySdn7NixGTJkSA4//PBccMEFre+rrKzc6hxf/OIX07t371x33XW55ZZbsmXLlvTr12+r8H/77bfTr1+/JElLS8s2r7+jPEUEAICy++1vf5va2tokSXV1dQ4//PAsW7YsjY2NOf/88zNq1Kg888wz2bJly/ue44ADDsiUKVPy6quv5umnn86gQYOyefPm/Nu//VuSZMGCBTn55JPT1NSUkSNHprq6Okny8ssv5/e//31hX4s72AAA7LBtrcHenk99/K+11ytXrszgwYNz5ZVXpnv37hk+fHhGjx6dbt265dBDD82WLVuydu3a9z1Pjx49ct111+Xiiy/OvHnzMnv27Nxwww2ZMWNGKioqMmvWrHTv3j3f+MY3cvnll+exxx7LwIED079//w/8tf+lLi1/fm+8E6ipqcnw4cPLPQYAQKezdOnSDBw4sNxjtLv3+7rfrzvb5Q72j370o9xxxx35wx/+kM997nO55pprsvfee2fChAl5+eWX07Xrn1aqjBw5MrfeemuSZPbs2XnwwQfT3Nyc008/PVOnTm2PUQvTtHZtmhsayj1Gp9a1e/dUfPzj5R4DAGArJQ/s1157Ld/97nfz4IMPZtCgQZk1a1auueaa3H///Vm8eHGeeuqp1r8k/S8//elP89hjj+WRRx5JU1NTzjjjjAwZMiSjRo0q9biFaW5oyMujx5Z7jE7twCcfL/cIAADvUfI/ctxvv/3ys5/9LP/tv/231NXVZf369enTp0/efvvttLS0vCeuk2TevHk57bTTsvvuu2ePPfbIhAkTWhehAwBAR9YuTxHZZZdd8txzz+ULX/hCHnnkkUyePDmvv/56evTokVNPPTWHHXZYpkyZktWrVyf50zMLBw0a1Hr8gAEDCvtseAAAKKV2e4rIiBEj8vLLL+eee+7JpEmTctlll+XAAw/MFVdckU984hP59re/nUsvvTT33HNP6uvrt3quYWVlZTZt2tTmay1atKgUX8J2+Uzv3uUeodNramrKkg7wswaAD4uWlpbU19eXe4x2V19fv1192W6B/V8PGD/33HNz9913p3///rntttta919yySU57LDDsnHjxvTs2XOrz4Ovr69PVVVVm681ePDg4gb/gBr+/7vxlE5FRUWH+FkDwIfF0qVL3/PhLh8GlZWV7/sUkW0p+RKR+fPn58ILL2x93dzcnMbGxixcuDA///nPW7c3Njama9euqaioyMCBA7Ns2bLWfX+5ZAQAADqqkt/BPuCAA3L55Zfnueeey8EHH5zZs2dnn332ycc//vHccMMN2WeffdKnT5985zvfyfHHH5/u3btnzJgxmTlzZo466qi0tLTkwQcfzKWXXlrqUQEA2EGlflTx9j6md/Hixbnpppvy4osvpmvXrjnooINy2WWXZd999y3ZjCUP7E996lO55ZZbcv3112f16tUZMWJEbr311uy2225Zvnx5/vEf/zEbN27MqFGjcu211yZJjj/++CxdujRf/epX09jYmAkTJuSYY44p9agAAOygUj+qeHse07tly5ZMmjQpZ599du666640Nzfn3nvvzdlnn51nnnkmPXr0KMmM7bIGe9SoUdt8hvXUqVPf9wNkzjvvvJx33nmlHg0AgE7qD3/4Q5YvX56xY8emouJP2XvuuefmjTfeSF1dXXbfffeSXLddHtMHAADtrW/fvhkyZEhOO+20fO9738uLL76YpqamXHvttSWL60RgAwDQif3gBz/IuHHj8tRTT+XUU0/N4Ycfnh/84AclvWa7PaYPAADa2y677JLJkydn8uTJqaury7/+67/m29/+dgYNGpQjjzyyJNd0BxsAgE5p3rx5OfXUU1tf9+7dO//4j/+Yww47LL/73e9Kdl2BDQBAp3TYYYdlyZIluf3227N+/fo0Njbm17/+dV588cWMHDmyZNe1RAQAgE5p1113zf33358ZM2bkhz/8YRobGzNo0KDcdNNN2W+//Up2XYENAEBhunbvvl3Pqv4g598e++23X+bMmVOiabZNYAMAUJjt+ZTFzsoabAAAKJDABgCAAglsAAAokMAGAIACCWwAACiQwAYAgAIJbAAAKJDnYAMAUJg/btycxqbmkp3/IxVd06uqR8nOXwSBDQBAYRqbmnP6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B2batGmZNGlS1q1bl3HjxmXChAkf5OsDAIB29b6Bveeee6aysjJ1dXVJstWa6CTZZZddcsstt/zNC3zta1/L1772tffdf+SRR25z+ymnnJJTTjnlb54fAAA6kr/6R46XX355kmTvvffOOeec0y4DAQDAzqxNTxE555xzsnjx4ixduvQ9H1lubTQAAPw/bQrs//W//lfuvPPO7Lbbbqmo+H+HdOnSRWADAMCfaVNgV1dX56677srIkSNLPQ8AAOzU2vSYvvr6+hx++OGlngUAAHZ6bQrs448/Pvfff3+pZwEAgJ1em5aILFu2LA8//HBuv/327Lrrrlvte+KJJ0oyGAAA7IzaFNjjx4/P+PHjSz0LAADs9NoU2F/+8pdLPQcAAHQKbQrsCRMmpEuXLtvcd9999xU6EAAA7MzaFNh/+Xi+urq6PPnkk/nKV75SkqEAAGBn1abA/trXvvaebSeffHKuuuqqnH/++YUPBQAAO6s2PaZvWz796U/nP//zP4ucBQAAdnptuoP9l4/i27JlS/71X/81n/3sZ0syFAAA7KzaFNjf/e53t3rdrVu3DBgwINdcc00pZgIAgJ1WmwL7Zz/7WannAACATqFNgZ0kL774Yqqrq/PWW2+lb9++GT9+fA455JBSzgYAADudNv2R4/z583PGGWeksbExBx10UJqbm/O1r30tTz75ZKnnAwCAnUqb7mDfdtttufXWWzNq1KjWbWPHjs3NN9+c0aNHl2w4AADY2bTpDvYbb7zxng+b+fu///ssX768JEMBAMDOqk2Bvccee+TZZ5/dattzzz2XPffcsyRDAQDAzqpNS0QmT56c888/P2PGjMkee+yR5cuX58knn8x3vvOdUs8HAAA7lTYF9nHHHZff//73qampyapVq9LU1JTvf//7GTFiRKnnAwCAnUqblog8/PDDueOOO3LRRRfl7rvvzgknnJDJkydn3rx5pZ4PAAB2Km0K7Lvuuiv33ntv9t133yTJKaeckrvvvjuzZs0q6XAAALCzaVNgr1mzJvvvv/9W2/bff/+8++67JRkKAAB2Vm0K7H322Sf/+3//7622PfTQQ613tAEAgD9p0x85XnnllTn33HPzwAMP5JOf/GTefvvtrF27Nt///vdLPR8AAOxU2hTYBx10UH7605/m3/7t37Jq1ap88pOfzBFHHJGPf/zjpZ4PAAB2Km0K7CTp3bt3xo8fX8pZAABgp9emNdgAAEDbCGwAACiQwAYAgAIJbAAAKJDABgCAAglsAAAokMAGAIACCWwAACiQwAYAgAIJbAAAKJDABgCAAglsAAAokMAGAIACCWwAACiQwAYAgAIJbAAAKJDABgCAAglsAAAokMAGAIACtWtgz5kzJ1dffXXr6yuuuCJ/93d/l6FDh2bo0KH5yle+0rrvoYceysiRIzN8+PBMnz49W7Zsac9RAQDgA2mXwG5oaMgtt9ySGTNmbLX99ddfzw9/+MO88MILeeGFF/LjH/84SfLKK69k5syZueeeezJ//vwsWrQoDz/8cHuMCgAAO6RdAnv69Ol59dVXc+qpp7Zu27JlS5YsWZJ99933Pe9//PHHc8IJJ2TvvfdOnz59MnHixFRXV7fHqAAAsEPaJbAvuuiizJkzJ7vuumvrtqVLl6Zbt26ZNm1aDj300Jx55plZunRpkqS2tjaDBg1qfe+AAQOyZMmS9hgVAAB2SEV7XKRfv37v2bZ+/foMGzYsl156aQYOHJjvfe97Oe+88zJv3rzU19ensrKy9b2VlZXZtGlTm6+3aNGiQubeEZ/p3bvcI3R6TU1NWdIBftZAMfzebB9+d0LptUtgb8uQIUNy9913t76eMmVK7r333tTW1qZnz57ZvHlz6776+vpUVVW1+dyDBw8udNYPomH16nKP0OlVVFR0iJ81UAy/N9uH351QnJqamm1uL9tj+v793/89jzzySOvr5ubmbNmyJd27d8/AgQNTW1vbuu8vl4wAAEBHVdbnYN944415/fXX09DQkBkzZmTfffdN//79M3r06MydOzeLFy9OXV1d5syZkzFjxpRzVAAAaJOyLRE59NBDM3Xq1EyaNCl1dXUZNmxYZs6cmSQ58MADM23atEyaNCnr1q3LuHHjMmHChHKNCgAAbdaugT1lypStXp9++uk5/fTTt/neU045Jaecckp7jAUAAIXxUekAAFAggQ0AAAUS2AAAUCCBDQAABRLYAABQIIENAAAFEtgAAFAggQ0AAAUS2AAAUCCBDQAABRLYAABQIIENAAAFEtgAAFAggQ0AAAUS2AAAUCCBDQAABRLYAABQIIENAAAFEtgAAFAggQ0AAAUS2AAAUCCBDQAABRLYAABQIIENAAAFEtgAAFAggQ0AAAUS2AAAUCCBDQAABRLYAABQIIENAAAFEtgAAFAggQ0AAAUS2AAAUCCBDQAABRLYAABQIIENAAAFEtgAAFAggQ0AAAUS2AAAUCCBDQAABRLYAABQIIENAAAFEtgAAFAggQ0AAAUS2AAAUCCBDQAABRLYAABQIIENAAAFatfAnjNnTq6++urW1/Pnz88xxxyToUOHZurUqdmwYUPrvoceeigjR47M8OHDM3369GzZsqU9RwUAgA+kXQK7oaEht9xyS2bMmNG6beXKlbn88stz00035dlnn01TU1PuuOOOJMkrr7ySmTNn5p577sn8+fOzaNGiPPzww+0xKgAA7JB2CTQqQx0AAA5USURBVOzp06fn1Vdfzamnntq67emnn84hhxySESNGpKqqKlOnTk11dXWS5PHHH88JJ5yQvffeO3369MnEiRNb9wEAQEfWLoF90UUXZc6cOdl1111bt9XW1mbQoEGtrwcOHJh33303dXV179k3YMCALFmypD1GBQCAHVLRHhfp16/fe7bV19end+/era979OiRLl26ZNOmTamvr09lZWXrvsrKymzatKnN11u0aNGODVyAz/zZ10ZpNDU1ZUkH+FkDxfB7s3343Qml1y6BvS2VlZVpaGhofb158+a0tLSkqqoqPXv2zObNm1v31dfXp6qqqs3nHjx4cKGzfhANq1eXe4ROr6KiokP8rIFi+L3ZPvzuhOLU1NRsc3vZHtM3cODA1NbWtr5eunRp+vbtm169er1n318uGQEAgI6qbIF99NFHZ8GCBVm4cGE2btyY2bNnZ8yYMUmS0aNHZ+7cuVm8eHHq6uoyZ86c1n0AANCRlW2JyCc/+cncfPPN+cY3vpF33nknI0eOzMUXX5wkOfDAAzNt2rRMmjQp69aty7hx4zJhwoRyjQoAAG3WroE9ZcqUrV4feeSROfLII7f53lNOOSWnnHJKO0wFAADF8VHpAABQIIENAAAFEtgAAFAggQ0AAAUS2AAAUCCBDQAABRLYAABQIIENAAAFEtgAAFAggQ0AAAUS2AAAUCCBDQAABRLYAABQIIENAAAFEtgAAFAggQ0AAAUS2AAAUCCBDQAABRLYAABQoIpyDwAAtJ/6HlVZ98f6co/RqX2komt6VfUo9xiUkcAGgA+RxuaW/NOMueUeo1N78NIvlXsEyswSEQAAKJDABgCAAglsAAAokMAGAIACCWwAACiQwAYAgAIJbAAAKJDABgCAAglsAAAokMAGAIACCWwAACiQwAYAgAIJbAAAKJDABgCAAglsAAAokMAGAIACCWwAACiQwAYAgAIJbAAAKJDABgCAAglsAAAokMAGAIACCWwAACiQwAYAgAIJbAAAKJDABgCAAglsAAAokMAGAIAClT2wZ82alc997nMZOnRohg4dmsMPPzxJMn/+/BxzzDEZOnRopk6dmg0bNpR5UgAA+NvKHtivv/56rr/++rzwwgt54YUX8uyzz2blypW5/PLLc9NNN+XZZ59NU1NT7rjjjnKPCgAAf1OHCOz99ttvq21PP/10DjnkkIwYMSJVVVWZOnVqqquryzQhAAC0XUU5L75x48a8+eabmTlzZl566aX0798/V199dWprazNo0KDW9w0cODDvvvtu6urq0rt37zJOTEdS36Mq6/5YX+4xOrWPVHRNr6oe5R4DAHYqZQ3sd999NyNGjMjZZ5+dgw46KI8++mgmTpyYL37xi1uFdI8ePdKlS5ds2rSpTeddtGhRqUZus8/4D4GSa2xuyT/NmFvuMTq1ey8aneXL/m+5x+BDwu9NOoumpsYsWlRb7jEoo7IG9l577ZX777+/9fUpp5yS++67LzU1Nenbt2/r9s2bN6elpSVVVVVtOu/gwYMLn3V7NaxeXe4RYIdVVHykQ/zviQ8HvzfpLPzu/PCoqanZ5vayrsFetGhR7r333q22NTQ05H/8j/+R2tra1m1Lly5N375906tXr3aeEAAAtk9ZA7tnz56ZOXNmFixYkC1btuT+++9PY2NjjjzyyCxYsCALFy7Mxo0bM3v27IwZM6acowIAQJuUdYnIwIEDc9NNN+Vb3/pW3n777ey333654447sscee+Tmm2/ON77xjbzzzjsZOXJkLr744nKOCgAAbVLWwE6S4447Lscdd9x7th955JE58sgj238gAADYAWV/DjYAAHQmAhsAAAoksAEAoEACGwAACiSwAQCgQAIbAAAKJLABAKBAAhsAAAoksAEAoEACGwAACiSwAQCgQAIbAAAKJLABAKBAAhsAAAoksAEAoEACGwAACiSwAQCgQAIbAAAKJLABAKBAAhsAAAoksAEAoEACGwAACiSwAQCgQAIbAAAKJLABAKBAAhsAAAoksAEAoEACGwAACiSwAQCgQBXlHgDouLpVNGbNxvpyj9Gpde/2kXy0x8fKPQYABRLYwPva0tKYKdXnl3uMTm32ybeXewQACmaJCAAAFEhgAwBAgQQ2AAAUSGADAECBBDYAABRIYAMAQIEENgAAFEhgAwBAgQQ2AAAUSGADAECBBDYAABRIYAMAQIEqyj0AAEBn0q2iMWs21pd7jE6te7eP5KM9PlbuMd6XwAYAKNCWlsZMqT6/3GN0arNPvr3cI/xVlogAAECBBDYAABRIYAMAQIEENgAAFKjDBnZNTU3GjRuXIUOG5Mwzz8yqVavKPRIAAPxNHTKwN23alKlTp+b888/PwoULM2DAgNx4443lHgsAAP6mDvmYvgULFqRv3745/vjjkyQXXXRRDj/88Kxfvz4f/ehHyzwdAAC8vw55B3vZsmUZNGhQ6+tevXqlV69eWbZsWRmnAgCAv61LS0tLS7mH+Eu333573njjja2WhRx99NH5zne+kxEjRvzVY2tqako9HgAAJEmGDx/+nm0dcolIZWVlGhoattq2adOm7LLLLn/z2G19kQAA0F465BKRgQMHpra2tvX1unXrsnbt2vTv3798QwEAQBt0yMA+9NBD89Zbb+WJJ55IQ0NDZs6cmVGjRrXpDjYAAJRTh1yDnSQvvfRSpk+fnmXLlmXYsGH5zne+k759+5Z7LAAA+Ks6bGADAMDOqEMuEQEAgJ2VwAYAgAIJbGgHNTU1GTduXIYMGZIzzzwzq1atKvdIADuFOXPm5Oqrry73GLBdBDaU2KZNmzJ16tScf/75WbhwYQYMGLDVhygB8F4NDQ255ZZbMmPGjHKPAtutQ37QDHQmCxYsSN++fXP88ccnSS666KIcfvjhWb9+fT760Y+WeTqAjmn69Ol55513cuqpp6axsbHc48B2cQcbSmzZsmUZNGhQ6+tevXqlV69eWbZsWRmnAujYLrroosyZMye77rpruUeB7SawocQ2btyYnj17brWtZ8+eqa+vL9NEAB1fv379yj0CfGACG0qssrIyDQ0NW23btGmTTyYFgE5KYEOJDRw4MLW1ta2v161bl7Vr16Z///7lGwoAKBmBDSV26KGH5q233soTTzyRhoaGzJw5M6NGjXIHGwA6KYENJdazZ8/ccccdueuuu3LIIYektrY21157bbnHAgBKpEtLS0tLuYcAAIDOwh1sAAAokMAGAIACCWwAACiQwAYAgAIJbAAAKJDABgCAAglsgA7u3XffzZVXXpkvfOELGTJkSI466qjcdNNNqa+vL/Q6c+fOzUknnVToOZPkiiuuyDe/+c3CzwvQUQlsgA7uoosuSnNzc5544om8+OKLueeee1JTU5P/+T//Z6HX+dKXvpTq6upCzwnwYSSwATq4l156Kccee2x69+6dJOnfv3+uuuqq9O3bN2+++Wb23XffrF69uvX9Z599dmbNmpXkT3ePp02blmOPPTZHHHFEpk6d+p67yWPGjMlPfvKTVFdX5/jjj09zc3O++MUvZt68ea3vWbVqVQ444IC8+eabaW5uzpw5c3Lsscfm85//fCZOnJjly5e3vvc3v/lNxo8fnyFDhuScc87JH/7wh1J+ewA6HIEN0MGNGTMmV199dW644YbMnz8/a9asydChQ3PllVe26fgFCxbkBz/4QebNm5fTTjstTz31VBoaGpIkr7zySt56660cd9xxre/v2rVrxo8fn7lz57ZumzdvXoYPH54999wz9913X370ox/lzjvvzK9//et87nOfy7nnnpumpqasWbMmkyZNyle/+tU8//zz+ad/+qf86le/KvYbAtDBCWyADu6GG27IlVdemWXLluWyyy7LYYcdllNPPTUvvfRSm44fNmxY9tprr3zsYx/LoYcemo9+9KP5xS9+kSR59NFHc/zxx6eqqmqrY0466aQ899xzWbNmTev7/mt99kMPPZTzzjsve++9d7p3754LLrggf/jDH7Jw4cL8/Oc/z6677pr//t//eyoqKnLkkUfmiCOOKPC7AdDxCWyADu6/7ijfddddef755/OTn/wke+yxR84666z88Y9//JvH77bbbq3/3qVLl3z5y1/OY489lqampjz++OM5+eST33PMXnvtlaFDh+aJJ57I66+/njfeeCP/8A//kCRZsWJFrr322owYMSIjRozIwQcfnI0bN2b58uVZvXp1dt99963Oteeee+7gdwBg5yKwATqwX/7ylxkxYkQ2bNiQ5E+xvf/+++fGG2/Mhg0bsnHjxiTJli1bWo+pq6v7q+f88pe/nF/84hd55plnsssuu2TEiBHbfN9JJ52UJ554IvPmzdvqLvfuu++em2++Oc8//3zrP9XV1Rk3blx23333rFixYqvzrFq16gN//QA7I4EN0IEdfPDB6dWrV66++uq88cYbSZI1a9bktttuy5577pkDDzwwH/vYxzJ37tw0NTXlmWeeyWuvvfZXz7nnnnvmoIMOyk033ZQvf/nL7/u+4447LosXL95qeUiSnHzyybn99tvzxhtvpKWlJdXV1TnxxBOzcuXKfPGLX8yGDRty9913p7GxMc8++2yeeeaZYr4ZADsJgQ3QgVVWVubBBx9MZWVlTj/99AwZMiRjxozJ8uXLc99996V79+655ppr8i//8i85+OCD8+ijj2bs2LF/87wnnXRS3n777YwfP/5931NVVZV/+Id/SM+ePbe6y3322Wfn2GOPzZlnnplhw4blvvvuy2233ZbPfOYz6d27d77//e/nqaeeysEHH5xZs2blmGOOKeR7AbCz6NLS0tJS7iEAAKCzcAcbAAAKJLABAKBAAhsAAAoksAEAoEACGwAACiSwAQCgQAIbAAAKJLABAKBAAhsAAAr0/wFomA4cWbZa/wAAAABJRU5ErkJggg==
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Most-people-who-survived-were-from-Southamption,-but-most-people-who-did-not-survive-also-boarded-from-Southamption.-Its-safe-to-say-the-majority-of-the-ship-came-from-Southamption.">Most people who survived were from Southamption, but most people who did not survive also boarded from Southamption. Its safe to say the majority of the ship came from Southamption.<a class="anchor-link" href="#Most-people-who-survived-were-from-Southamption,-but-most-people-who-did-not-survive-also-boarded-from-Southamption.-Its-safe-to-say-the-majority-of-the-ship-came-from-Southamption.">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="How-do-the-Age-values-match-with-different-passenger-class">How do the Age values match with different passenger class<a class="anchor-link" href="#How-do-the-Age-values-match-with-different-passenger-class">&#182;</a></h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[29]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sns</span><span class="o">.</span><span class="n">violinplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="s1">&#39;Pclass&#39;</span><span class="p">,</span><span class="n">y</span><span class="o">=</span><span class="s2">&quot;Age&quot;</span><span class="p">,</span><span class="n">data</span><span class="o">=</span><span class="n">titanic_df</span><span class="p">)</span>
<span class="n">sns</span><span class="o">.</span><span class="n">swarmplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="s1">&#39;Pclass&#39;</span><span class="p">,</span><span class="n">y</span><span class="o">=</span><span class="s2">&quot;Age&quot;</span><span class="p">,</span><span class="n">data</span><span class="o">=</span><span class="n">titanic_df</span><span class="p">,</span><span class="n">color</span><span class="o">=</span><span class="s1">&#39;0.2&#39;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[29]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x1284e5898&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAtIAAAHlCAYAAADY/RsiAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjAsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+17YcXAAAgAElEQVR4nOzdd3Sc530n+u/bpndUohd2UoWiukOLLoqKHdtXXjurOHvsTezdnJyNnLVzjo/j9XXu7s2uEye5d+/6xMnaki3SkosoSrJESaQoikUkSIpgAQGCBAt6BwYzg+nlfe8fQww1HJBoU4TB93OOjzQP3nnf31iYmS+e9ymCpmkaiIiIiIhoQcRCF0BEREREtBwxSBMRERERLQKDNBERERHRIjBIExEREREtAoM0EREREdEiMEgTERERES2CXOgCFqO1tbXQJRARERHRCrF169ZZ25dlkAZu/YKIiIiIiLLldh24HNpBRERERLQIDNJERERERIvAIE1EREREtAgM0kREREREi8AgTURERES0CAzSRERERESLwCBNRERERLQIDNJERERERIvAIE1EREREtAgM0kREREREi8AgTURERES0CAzSRERERESLwCBNRERERLQIDNJERERERIvAIE1EREREtAgM0kREREREi8AgTURERES0CAzSlFfXrl3D3/393yMWixW6FCIiIqIlYZCmvDp+/DjeP3IE09PThS6FiIiIaEkYpCmvNE0DACQSiQJXQkRERLQ0DNJUEAzSREREtNwxSFNBMEgTERHRcscgTQXBIE1ERETLHYM0FQSDNBERES13DNKUVzOTDbn8HRERES13DNKUV/F4HAB7pImIiGj5Y5CmvJoJ0jP/JCIiIlquGKQpr2YCNId2EBER0XLHIE15FY0mAzR7pImIiGi5Y5CmvIrGogDYI01ERETLH4M05VU0wiBNRERExYFBmvIqGmWQJiIiouLAIE15FbkepGcCNREREdFyxSBNeTUz2ZBBmoiIiJY7BmnKq0gkAoBBmoiIiJY/BmnKqyiHdhAREVGRYJCmvIpGkz3SMz3TRERERMsVgzTlVTQahSDpGKSJiIho2WOQpryKxWIQJD2DNBERES17DNKUV7FoFIKkRzAULnQpREREREvCIE15k0gkkEjEIUh6hMPskSYiIqLljUGa8mZm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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="The-mean-ages-for-class-1-are-higher-than-class-2,-which-is-higher-than-class-3.-This-is-intuitive-becasue-richer-people-tend-to-be-of-older-age.-Class-1-fare-is-the-most-expensive-so-has-a-higher-mean-age.">The mean ages for class 1 are higher than class 2, which is higher than class 3. This is intuitive becasue richer people tend to be of older age. Class 1 fare is the most expensive so has a higher mean age.<a class="anchor-link" href="#The-mean-ages-for-class-1-are-higher-than-class-2,-which-is-higher-than-class-3.-This-is-intuitive-becasue-richer-people-tend-to-be-of-older-age.-Class-1-fare-is-the-most-expensive-so-has-a-higher-mean-age.">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Putting-it-together,-We-can-plot-for-survived/Not-Survived-for-Males/Females-by-Age-Group-and-Fare-paid-for-the-ship.">Putting it together, We can plot for survived/Not Survived for Males/Females by Age Group and Fare paid for the ship.<a class="anchor-link" href="#Putting-it-together,-We-can-plot-for-survived/Not-Survived-for-Males/Females-by-Age-Group-and-Fare-paid-for-the-ship.">&#182;</a></h2><h4 id="0---Not-Survived">0 - Not Survived<a class="anchor-link" href="#0---Not-Survived">&#182;</a></h4><h4 id="1---Survived">1 - Survived<a class="anchor-link" href="#1---Survived">&#182;</a></h4><h3 id="Red-circles-are-Male-while-Blue-circle-markers-are-Female">Red circles are Male while Blue circle markers are Female<a class="anchor-link" href="#Red-circles-are-Male-while-Blue-circle-markers-are-Female">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[33]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sns</span><span class="o">.</span><span class="n">scatterplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="s2">&quot;Age&quot;</span><span class="p">,</span> <span class="n">y</span><span class="o">=</span><span class="s2">&quot;Fare&quot;</span><span class="p">,</span>
                      <span class="n">hue</span><span class="o">=</span><span class="s2">&quot;Sex&quot;</span><span class="p">,</span> <span class="n">data</span><span class="o">=</span><span class="n">titanic_df</span><span class="p">,</span><span class="n">palette</span><span class="o">=</span><span class="s2">&quot;Set1&quot;</span><span class="p">,</span> <span class="n">size</span><span class="o">=</span><span class="s2">&quot;Survived&quot;</span><span class="p">)</span>
<span class="n">sns</span><span class="o">.</span><span class="n">set_style</span><span class="p">(</span><span class="s2">&quot;ticks&quot;</span><span class="p">,</span> <span class="p">{</span><span class="s2">&quot;xtick.major.size&quot;</span><span class="p">:</span> <span class="mi">12</span><span class="p">,</span> <span class="s2">&quot;ytick.major.size&quot;</span><span class="p">:</span> <span class="mi">12</span><span class="p">})</span>
<span class="n">sns</span><span class="o">.</span><span class="n">set_context</span><span class="p">(</span><span class="s2">&quot;paper&quot;</span><span class="p">,</span> <span class="n">font_scale</span><span class="o">=</span><span class="mf">1.4</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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AAAA6DKvv/66MjIy9NBDD8k0TZWVlWn8+PEqKirqdB/Dhw/X8OHDU1hlxwjQAAAA6DLFxcVavny5jjrqKB1yyCFavHixrrvuOp177rnaf//9demll+qqq67S/Pnz1djYqOLiYlVUVOjxxx+XaZqaNm2aZs+erX//+98KhUKaMmWKRo8erZtuuklHH320mpubtWDBAknSZZddpsMPP1zXXXed1q1bpz59+igra/e34SNAAwAAoMuMGDFCM2fO1AMPPKDy8nJNmjRJpmnu8HMnT56sCRMm6Nprr9X69etVW1urcePGtV4/6aST9MILL2j06NFavXq1rrzySv3sZz/TY489JsMwdP755ysQCCgSieixxx7Tc889p2XLlu3218AcaAAAAHSZdevW6ZBDDtHChQv1r3/9S5988oneeOON1uvfPCR74MCBkqSJEyfqhRde0PPPP6+JEye2Xh8zZozWrFmjtWvX6sADD1RdXZ0+//xznXfeebrwwgtVV1enTZs2adiwYZKkUaNGJeVrIEADAACgyyxfvlwLFy6UJGVnZ6tfv34KBAKqra1VIpHQhg0bWj/X6dweVQ8//HCtWrVKmzZtUmlpaZvro0aN0t13362JEycqPz9fgwYN0sMPP6wHH3xQEydO1JAhQ7R27VpJ0kcffZSUr4EpHAAAAOgy06dP19y5c3XKKafI7/drzJgxmjdvnn7729+qb9++Ki4ubncbl8ulffbZZ4cLDSdOnKjZs2e3ji6feeaZmjFjhsLhsKZNm6b9999fAwcOVFlZmfbee295PJ7d/hoc1jfHybu50tJSVVRUpLsMAACAPcaGDRs0aNCgdJeRVjv6HnSUO5nCAQAAANhAgAYAAABsIEADAAAANhCgAQAAABsI0AAAAIANBGgAAADABvaBBgAAQKdUVlaqsrJSAwYM0IABA3apj0QioTlz5qiqqkqjR4/WNddck+QqU48ADQAAgA41NTVp9uzZ2rhxo+LxuDwej/bdd1/ddtttCgQCtvp66aWXVFpaqttvv12/+tWvtHbtWo0ePTpFlacGUzgAAADQodmzZ2vt2rVqaGhQKBRSQ0OD1q5dq1mzZtnua/Xq1Ro3bpwk6YgjjtB7772X7HJTjgANAACA71RZWamNGzfKNM027aZpauPGjaqsrLTVXzAYlN/vlyT5fD6FQqGk1dpVCNAAAAD4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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Preparing-data-for-Logistic-Regression">Preparing data for Logistic Regression<a class="anchor-link" href="#Preparing-data-for-Logistic-Regression">&#182;</a></h1><h3 id="We-need-to-convert-string-values-into-binary-(0-or-1)-values">We need to convert string values into binary (0 or 1) values<a class="anchor-link" href="#We-need-to-convert-string-values-into-binary-(0-or-1)-values">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[34]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Converting the Embarked column into a numerical binary value for Q,S and C. If both Q and C are 0, </span>
<span class="c1"># then the value would automatically be C</span>

<span class="n">embarked</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">get_dummies</span><span class="p">(</span><span class="n">titanic_df</span><span class="p">[</span><span class="s1">&#39;Embarked&#39;</span><span class="p">],</span> <span class="n">drop_first</span><span class="o">=</span><span class="s1">&#39;True&#39;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[35]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">embarked</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[35]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Q</th>
      <th>S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[36]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Converting the P assenger Class column into a numerical binary value for 1,2,3. If both 2 and 3 are 0, </span>
<span class="c1"># then the value would automatically be class 1</span>

<span class="n">pcl</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">get_dummies</span><span class="p">(</span><span class="n">titanic_df</span><span class="p">[</span><span class="s1">&#39;Pclass&#39;</span><span class="p">],</span> <span class="n">drop_first</span><span class="o">=</span><span class="s1">&#39;True&#39;</span><span class="p">)</span>
<span class="n">pcl</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[36]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[37]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Converting Sec column to a binary. If Male = 0, then the value would be a female automatically</span>
<span class="n">sex</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">get_dummies</span><span class="p">(</span><span class="n">titanic_df</span><span class="p">[</span><span class="s1">&#39;Sex&#39;</span><span class="p">],</span> <span class="n">drop_first</span><span class="o">=</span><span class="s1">&#39;True&#39;</span><span class="p">)</span>
<span class="n">sex</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[37]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[40]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Combining the above dataframe to our titanic dataframe</span>
<span class="n">titanic_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">concat</span><span class="p">([</span><span class="n">titanic_df</span><span class="p">,</span> <span class="n">embarked</span><span class="p">,</span> <span class="n">pcl</span><span class="p">,</span> <span class="n">sex</span><span class="p">],</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[41]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[41]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Q</th>
      <th>S</th>
      <th>2</th>
      <th>3</th>
      <th>male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>S</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>S</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>S</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>S</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[45]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df_binary</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="p">[[</span><span class="s2">&quot;Survived&quot;</span><span class="p">,</span><span class="s2">&quot;SibSp&quot;</span><span class="p">,</span><span class="s2">&quot;Parch&quot;</span><span class="p">,</span><span class="s2">&quot;Fare&quot;</span><span class="p">,</span><span class="s2">&quot;male&quot;</span><span class="p">,</span><span class="s2">&quot;Q&quot;</span><span class="p">,</span><span class="s2">&quot;S&quot;</span><span class="p">,</span><span class="mi">2</span><span class="p">,</span><span class="mi">3</span><span class="p">,</span><span class="s2">&quot;male&quot;</span><span class="p">]]</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[46]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Final dataset for Regression</span>
<span class="n">df_binary</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[46]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>male</th>
      <th>Q</th>
      <th>S</th>
      <th>2</th>
      <th>3</th>
      <th>male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Logistic-Regression">Logistic Regression<a class="anchor-link" href="#Logistic-Regression">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Performing-Machine-Learning-on-the-dataset-prepared--">Performing Machine Learning on the dataset prepared -<a class="anchor-link" href="#Performing-Machine-Learning-on-the-dataset-prepared--">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[53]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Assigning dependant and independant variables </span>

<span class="c1"># Survived column is our dependant variable. We are trying to predict this variable</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">df_binary</span><span class="p">[</span><span class="s1">&#39;Survived&#39;</span><span class="p">]</span>

<span class="c1"># The other columns are out independant variables. Hence we will drop Survived column from the dataframe</span>
<span class="n">X</span> <span class="o">=</span> <span class="n">df_binary</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="s1">&#39;Survived&#39;</span><span class="p">,</span> <span class="n">axis</span> <span class="o">=</span> <span class="mi">1</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[54]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span><span class="n">y</span><span class="p">,</span><span class="n">test_size</span><span class="o">=</span><span class="mf">0.2</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[57]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">classifier</span> <span class="o">=</span> <span class="n">LogisticRegression</span><span class="p">()</span>
<span class="n">classifier</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[57]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class=&#39;warn&#39;, n_jobs=None, penalty=&#39;l2&#39;,
                   random_state=None, solver=&#39;warn&#39;, tol=0.0001, verbose=0,
                   warm_start=False)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[58]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span> <span class="n">classifier</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stderr output_text">
<pre>/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to &#39;lbfgs&#39; in 0.22. Specify a solver to silence this warning.
  FutureWarning)
</pre>
</div>
</div>

<div class="output_area">

    <div class="prompt output_prompt">Out[58]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class=&#39;warn&#39;, n_jobs=None, penalty=&#39;l2&#39;,
                   random_state=None, solver=&#39;warn&#39;, tol=0.0001, verbose=0,
                   warm_start=False)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[59]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">f</span><span class="s2">&quot;Training Data Score: {classifier.score(X_train, y_train)}&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">f</span><span class="s2">&quot;Testing Data Score: {classifier.score(X_test, y_test)}&quot;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Training Data Score: 0.7978910369068541
Testing Data Score: 0.7482517482517482
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[60]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Predict</span>
<span class="n">predictions</span> <span class="o">=</span> <span class="n">classifier</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
<span class="n">predictions</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[60]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>array([1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
       0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
       1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0,
       1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0,
       0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1,
       0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1,
       0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0])</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[61]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Comparison of our prediction with actual result</span>
<span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">({</span><span class="s2">&quot;Prediction&quot;</span><span class="p">:</span> <span class="n">predictions</span><span class="p">,</span> <span class="s2">&quot;Actual&quot;</span><span class="p">:</span> <span class="n">y_test</span><span class="p">})</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[61]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Prediction</th>
      <th>Actual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>689</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>279</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>508</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>496</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>150</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>474</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>469</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>794</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>864</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>553</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>226</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>204</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>713</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>751</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>349</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>74</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>321</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>743</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>873</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>647</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>327</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>684</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>769</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>91</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>272</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>770</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>141</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>733</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>741</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>636</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>672</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>345</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>68</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>357</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>514</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>81</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>231</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>881</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>174</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>188</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>419</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>319</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>876</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>808</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>706</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>534</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>554</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>90</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>608</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>869</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>148</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>666</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>582</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>236</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>780</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>884</th>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>143 rows × 2 columns</p>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[65]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="k">import</span> <span class="n">classification_report</span><span class="p">,</span> <span class="n">accuracy_score</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[64]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">predictions</span><span class="p">,</span> <span class="n">y_test</span><span class="p">))</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>              precision    recall  f1-score   support

           0       0.80      0.78      0.79        87
           1       0.67      0.70      0.68        56

    accuracy                           0.75       143
   macro avg       0.74      0.74      0.74       143
weighted avg       0.75      0.75      0.75       143

</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Accuracy-of-our-model">Accuracy of our model<a class="anchor-link" href="#Accuracy-of-our-model">&#182;</a></h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[66]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">accuracy_score</span><span class="p">(</span><span class="n">predictions</span><span class="p">,</span><span class="n">y_test</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[66]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>0.7482517482517482</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span> 
</pre></div>

    </div>
</div>
</div>

</div>
    </div>
  </div>
</body>

 


</html>
