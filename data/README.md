# Data

This directory should contain task data for OTora evaluation.

## Expected Structure

```
data/
├── webshop/
│   ├── train.csv          # WebShop training tasks
│   └── val.csv            # WebShop validation tasks
├── injecagent/
│   ├── email_train.json   # Email agent tasks
│   ├── email_val.json
│   ├── os_train.json      # OS agent tasks
│   └── os_val.json
└── README.md
```

## Setup

### WebShop
1. Clone the [WebShop](https://github.com/princeton-nlp/WebShop) repository
2. Follow their setup instructions to prepare the environment
3. Place the preprocessed task CSV files under `data/webshop/`

### InjecAgent (Email & OS)
1. Download data from [InjecAgent](https://github.com/DongqiShen/InjecAgent)
2. Place JSON files under `data/injecagent/`

Preprocessed splits used in our experiments will be released upon paper acceptance.
