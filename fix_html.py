import re

with open('windowed_sv_explorer.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Add null checks to updateDisplay
code = code.replace(
    'document.getElementById(sv1-val-).textContent = svRow[0].toExponential(3);',
    '''const el_sv1 = document.getElementById(sv1-val-);
    if(el_sv1) el_sv1.textContent = svRow[0].toExponential(3);'''
)

code = code.replace(
    'document.getElementById(sv-ratio-).textContent = ratio2;',
    '''const el_sv2 = document.getElementById(sv-ratio-);
    if(el_sv2) el_sv2.textContent = ratio2;'''
)

code = code.replace(
    'document.getElementById(sv3-ratio-).textContent = ratio3;',
    '''const el_sv3 = document.getElementById(sv3-ratio-);
    if(el_sv3) el_sv3.textContent = ratio3;'''
)

# Add robust try-catch to everything inside JS
js_inject = '''
window.onerror = function(msg, url, lineNo, columnNo, error) {
  const errDiv = document.createElement('div');
  errDiv.style = "position:fixed; top:10px; right:10px; background:rgba(255,0,0,0.9); color:white; padding:10px; z-index:9999; max-width:400px; word-wrap:break-word;";
  errDiv.innerHTML = "<b>JS Error:</b> " + msg + "<br>Line: " + lineNo + " Col: " + columnNo + "<br><pre>" + (error && error.stack ? error.stack : "") + "</pre>";
  document.body.appendChild(errDiv);
  return false;
};
'''
code = code.replace('<script>', '<script>' + js_inject)

with open('windowed_sv_explorer.py', 'w', encoding='utf-8') as f:
    f.write(code)
