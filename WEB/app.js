// Inicializar los iconos de Lucide
lucide.createIcons();

// Lógica de navegación tipo Single Page Application (SPA)
document.addEventListener('DOMContentLoaded', () => {
  const navLinks = document.querySelectorAll('.nav-link');
  const pages = document.querySelectorAll('.page');

  // Función para cambiar de página
  function navigateTo(targetId) {
    // Esconder todas las páginas
    pages.forEach(page => {
      page.classList.remove('active');
    });

    // Quitar active de los links del navbar principal
    document.querySelectorAll('.navbar .nav-link').forEach(link => {
      link.classList.remove('active');
      // Si el link apunta al target actual, se activa
      if (link.getAttribute('data-target') === targetId) {
        link.classList.add('active');
      }
    });

    // Mostrar página destino
    const targetPage = document.getElementById(targetId);
    if (targetPage) {
      targetPage.classList.add('active');
      // Scroll hacia arriba al cambiar de pestaña
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }

  // Asignar eventos de click a cualquier elemento con la clase .nav-link
  navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const target = link.getAttribute('data-target');
      if (target) {
        navigateTo(target);
      }
    });
  });

  // Animación de hover para botones tipo Duolingo (se maneja en CSS, pero agregamos soporte click)
  const buttons = document.querySelectorAll('.btn-primary');
  buttons.forEach(btn => {
    btn.addEventListener('mousedown', () => {
      btn.style.transform = 'translateY(2px)';
      btn.style.boxShadow = '0 2px 0 #050824';
    });
    btn.addEventListener('mouseup', () => {
      btn.style.transform = 'translateY(-2px)';
      btn.style.boxShadow = '0 6px 0 #050824';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.transform = '';
      btn.style.boxShadow = '';
    });
  });

  // Lógica de Predicción de Sueldo
  const btnPredict = document.getElementById('btn-predict');
  if (btnPredict) {
    btnPredict.addEventListener('click', () => {
      const category = document.getElementById('salary-category').value;
      const experience = document.getElementById('salary-experience').value;
      const resultContainer = document.getElementById('salary-result');
      const predictedValue = document.getElementById('predicted-value');

      if (!category || !experience) {
        alert('Por favor selecciona una categoría y nivel de experiencia para la predicción.');
        return;
      }

      // Lógica de simulación del modelo
      let baseSalary = 0;
      switch (category) {
        case 'tech': baseSalary = 3000; break;
        case 'data': baseSalary = 3200; break;
        case 'design': baseSalary = 2500; break;
        case 'marketing': baseSalary = 2200; break;
      }

      let multiplier = 1;
      switch (experience) {
        case 'junior': multiplier = 1; break;
        case 'mid': multiplier = 1.6; break;
        case 'senior': multiplier = 2.4; break;
      }

      const finalSalary = Math.round(baseSalary * multiplier);
      
      // Mostrar con un pequeño retraso para simular carga del modelo
      predictedValue.textContent = 'Calculando...';
      resultContainer.style.display = 'block';
      
      setTimeout(() => {
        predictedValue.textContent = `S/ ${finalSalary.toLocaleString('es-PE')}`;
      }, 600);
    });
  }
});
