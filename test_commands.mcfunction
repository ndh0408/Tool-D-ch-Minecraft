# Test .mcfunction file for JSON text component translation
# Comments should not be translated

# Simple tellraw command
tellraw @a {"text":"Welcome to the server!","color":"gold"}

# Title command
title @a title {"text":"Server Event","bold":true}

# Multiple text components
tellraw @a [{"text":"Player ","color":"gray"},{"text":"has joined the game","color":"green"}]

# Nested components with hover
tellraw @a {"text":"Click here for help","color":"aqua","clickEvent":{"action":"run_command","value":"/help"},"hoverEvent":{"action":"show_text","contents":{"text":"Get help information"}}}

# Command without JSON (should not be modified)
gamemode survival @a

# Another tellraw
tellraw @a {"text":"Thank you for playing!"}
