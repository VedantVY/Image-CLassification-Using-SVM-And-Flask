<!DOCTYPE html>
<html lang="en" data-bs-theme = "dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-YvpcrYf0tY3lHB60NNkmXc5s9fDVZLESaAA55NDzOxhy9GkcIdslK1eN7N6jIeHz" crossorigin="anonymous">

    </script>
    <title>MyMLModel</title>
</head>
<body>
    <nav class="navbar bg-body-tertiary">
        <div class="container-fluid">
          <a class="navbar-brand" href="#">
            <img src="/static/New.png" alt="Logo" width="30" height="30" class="d-inline-block align-text-top">
            FindMyFruit.ai
          </a>
        </div>
    </nav>
    <div class = "d-flex justify-content-center align-items-center m-5 ">
        <form action = "{{ url_for('predict') }}" class="row g-3 border p-3 rounded" method = "POST" enctype="multipart/form-data">
            <div class="mb-3">
                <label for="formFile" class="form-label" required>Add an image of the fruit</label>
                <input class="form-control" type="file" id="formFile" name = "Image">
            </div>
            <div class="col-auto">
                <button type="submit" class="btn btn-success mb-3">Find Fruit</button>
            </div>
        </form>
    </div>
    <div class = "d-flex justify-content-center align-items-center m-5" <?php if (empty($_GET)){ echo 'style="display:none;"'; } ?>>
    <div class="card" style="width: 18rem;">
        <img src="data:image/jpeg;base64,{{ img }}" class="card-img-top" alt="...">
        <div class="card-body">
          <p class="card-text">
{{ First }}: {{ FirstP }}%
{{ Second }}: {{SecondP}}%
The fruit is  {{ First }} with a probability of {{ FirstP }}%.
          </p>
        </div>
    </div>
    </div>  
</body>
</html>