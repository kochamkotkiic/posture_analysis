const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');

function createWindow () {
  const win = new BrowserWindow({
    width: 1200, // Domyślna szerokość w razie wyjścia z pełnego ekranu
    height: 800,
    transparent: true,
    frame: false,
    resizable: true,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    }
  });

  // MAGIA: Uruchom aplikację od razu na pełnym ekranie!
  win.maximize();

  win.loadFile('ui/index.html');

  // Obsługa klasycznych przycisków (Minimalizuj, Maksymalizuj, Zamknij)
  ipcMain.on('window-controls', (event, command) => {
    if (command === 'minimize') win.minimize();
    if (command === 'maximize') {
        if (win.isMaximized()) win.restore();
        else win.maximize();
    }
    if (command === 'close') win.close();
  });

  // Obsługa przełączania między pełnym ekranem a małym kotkiem
  ipcMain.on('resize-window', (event, width, height, isOverlay) => {
    win.setAlwaysOnTop(isOverlay, 'screen-saver');

    if (isOverlay) {
        if (win.isMaximized()) win.restore(); // Wyłączamy pełny ekran dla kotka
        win.setResizable(false);
        win.setSize(width, height);
    } else {
        win.setResizable(true);
        win.maximize(); // Wracając do kamery, znów rzucamy na pełny ekran!
    }
  });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});