  1. Set API Key as Environment Variable (Recommended)

  # On the remote server, add to ~/.bashrc or ~/.zshrc
  export WANDB_API_KEY=your_api_key_here

  # Or set it for just this session
  export WANDB_API_KEY=your_api_key_here

  2. Alternative: Use wandb login with --host flag

  # If you can't open browser on remote server
  wandb login --host
  # This gives you a URL to open on your local machine

  3. Install W&B on Remote Server

  # On the remote GPU server
  uv add wandb

  4. Run Training with W&B

  # Make sure API key is set, then run
  uv run python train.py \
    --train_data ./data/tokens/TinyStoriesV2-GPT4-train.bin \
    --val_data ./data/tokens/TinyStoriesV2-GPT4-train.bin \
    --wandb \
    --wandb_project cs336-transformer \
    --device cuda \
    --batch_size 64 \
    --max_iters 10000

  5. Optional: Run in Background

  # Use nohup to keep training running if you disconnect
  nohup uv run python train.py --wandb --wandb_project cs336-transformer ... > training.log 2>&1 &

  # Or use screen/tmux
  screen -S training
  uv run python train.py --wandb ...
  # Ctrl+A, D to detach

  6. Monitor from Your Local Machine

  - Go to https://wandb.ai
  - View real-time training progress
  - Download logs/checkpoints later

  The W&B dashboard will work from anywhere - you train on remote GPU but monitor on your local browser.